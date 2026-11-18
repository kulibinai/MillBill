"""CNC comments and controller-aware postprocessing (no toolpath generation)."""
import importlib
import json
from pathlib import Path
import re


def code_body(line, *, allow_nested=True):
    """Balanced lexical stripping; nested mode is diagnostic, not LinuxCNC syntax."""
    depth, body = 0, []
    for ch in line:
        if ch == ';' and depth == 0:
            break
        if ch == '(':
            if depth and not allow_nested:
                raise ValueError('nested comments are not LinuxCNC executable syntax')
            depth += 1
        elif ch == ')':
            if depth == 0:
                raise ValueError('unmatched CNC comment closing parenthesis')
            depth -= 1
        elif depth == 0:
            body.append(ch)
    if depth:
        raise ValueError('unclosed CNC comment')
    return ''.join(body)


def validate_linuxcnc_comments(text):
    for line in text.splitlines():
        code_body(line, allow_nested=False)


def flat_comment_label(label):
    return str(label).replace('(', '[').replace(')', ']')


def normalize_comments(text):
    """Flatten only balanced nested comment text; preserve code projection exactly."""
    result = []
    for line in text.splitlines(keepends=True):
        expected = code_body(line)
        depth, chars, semicolon = 0, [], False
        for ch in line:
            if semicolon:
                chars.append(ch)
                continue
            if ch == ';' and depth == 0:
                semicolon = True
            if ch == '(':
                chars.append('[' if depth else '(')
                depth += 1
            elif ch == ')':
                depth -= 1
                chars.append(']' if depth else ')')
            else:
                chars.append(ch)
        flattened = ''.join(chars)
        assert code_body(flattened, allow_nested=False) == expected
        result.append(flattened)
    return ''.join(result)


def safe_comment_label(label):
    return ' '.join(''.join(' ' if c in '()' or ord(c) < 32 or ord(c) == 127 else c
                            for c in str(label)).split())


def custom_program_lines(text):
    """Flatten balanced comments before FreeCAD Custom makes Path.Commands.

    Keep source files and executable code unchanged; reject malformed input.
    """
    text = normalize_comments(text)
    validate_linuxcnc_comments(text)
    return [line.strip() for line in text.splitlines() if line.strip()]


def sanitize_operation_headers(ops):
    """Repair only the typed label comment made by FreeCAD Op.Base.execute.

    Keep public operation identity and every numeric command unchanged. A
    malformed arbitrary comment must still fail the normal export lexer.
    """
    repaired = []
    for op in ops:
        commands = list(op.Path.Commands)
        expected = '(' + str(op.Label).replace('(', '') + ')'
        safe = '(' + safe_comment_label(op.Label) + ')'
        if commands and commands[0].Name == expected and expected != safe:
            assert not commands[0].Parameters
            path = importlib.import_module('Path')
            commands[0] = path.Command(safe)
            op.Path = path.Path(commands)
            repaired.append(op.Name)
    return repaired


def executable_state(text, request):
    wanted = {label: o for o in request['operations'] for label in (o['label'], flat_comment_label(o['label']))}
    controllers = {c['tc']: c for c in request['controllers']}
    selected = pending = speed = spindle = active = None
    seen = {}
    for line in text.splitlines():
        body = code_body(line)  # validate even label-only lines before reading metadata
        match = re.fullmatch(r'\s*\(begin operation: (.*)\)\s*', line) if not body.strip() else None
        if match:
            active = match[1] if match[1] in wanted else None
        if not body.strip() and '(finish operation:' in line:
            active = None
        words = [(k.upper(), float(v)) for k, v in re.findall(r'([A-Za-z])\s*([-+]?\d*\.?\d+)', body)]
        values = dict(words)
        if 'T' in values:
            assert values['T'].is_integer()
            pending = int(values['T'])
        if 'S' in values:
            speed = values['S']
        for k, v in words:
            if k == 'M' and v == 6:
                assert pending is not None
                selected = pending
            if k == 'M' and v in (3, 4, 5):
                spindle = int(v)
        # Coordinate blocks include modal moves and canned-cycle holes; they
        # must not escape the tool/spindle gate merely by omitting G0..G3.
        if active and any(k in values for k in 'XYZ'):
            op = wanted[active]
            c = controllers[op['tc']]
            assert selected == op['tool_number'] and spindle == c.get('spindle_command', 3)
            assert speed is not None and abs(speed-c['spindle_rpm']) < 1e-6
            seen[op['label']] = {'T': selected, 'M': spindle, 'S': speed}
    assert set(seen) == {o['label'] for o in request['operations']}
    return seen


def export_job(job, ops, post_name, out_path):
    """Modern Job Factory or legacy script with real TC objects; never bare ops."""
    try:
        processor = importlib.import_module('Path.Post.Processor')
    except ModuleNotFoundError as exc:
        if exc.name not in ('Path.Post.Processor', 'Path.Post'):
            raise
        processor = None
    original = job.Operations.Group
    old_args, old_output = job.PostProcessorArgs, job.PostProcessorOutputFile
    try:
        sanitize_operation_headers(ops)
        job.Operations.Group = list(ops)
        job.PostProcessorArgs, job.PostProcessorOutputFile = '--no-show-editor', str(out_path)
        factory = getattr(processor, 'PostProcessorFactory', None)
        if factory is not None:
            pp = factory.get_post_processor(job, post_name)
            if pp is None:
                raise RuntimeError('postprocessor unavailable')
            sections = pp.export()
            if not isinstance(sections, (list, tuple)) or len(sections) != 1 or not isinstance(sections[0][1], str):
                raise RuntimeError('single nonempty Job export required')
            text = sections[0][1]
        else:
            # The legacy script API accepts Path-bearing objects, including the
            # official controller Paths. No manual M6/M3 synthesis/fallback.
            postables, previous = [], None
            for op in ops:
                tc = op.ToolController
                if tc is not previous:
                    postables.append(tc)
                    previous = tc
                postables.append(op)
            legacy = importlib.import_module('Path.Post.scripts.%s_post' % post_name)
            text = legacy.export(postables, '-', '--no-show-editor')
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError('empty postprocessor output')
        request = {'operations': [], 'controllers': []}
        for index, op in enumerate(ops):
            tc = op.ToolController
            direction = str(getattr(tc, 'SpindleDir', 'Forward'))
            expected_m = 4 if direction == 'Reverse' else 3
            request['operations'].append({'label': op.Label, 'tc': index, 'tool_number': int(tc.ToolNumber)})
            request['controllers'].append({'tc': index, 'spindle_rpm': float(getattr(tc.SpindleSpeed, 'Value', tc.SpindleSpeed)),
                                           'spindle_command': expected_m})
        try:
            text = normalize_comments(text)
        except ValueError as error:
            # Quarantined producer bytes are evidence, never a published NC.
            raw = Path(str(out_path) + '.rejected.raw.txt')
            raw.write_text(text)
            bad = []
            for number, line in enumerate(text.splitlines(), 1):
                try:
                    code_body(line)
                except ValueError as line_error:
                    bad.append({'line': number, 'text': line, 'error': str(line_error)})
            Path(str(out_path) + '.rejected.json').write_text(json.dumps(
                {'stage': 'normalize_comments', 'error': str(error), 'raw_file': str(raw),
                 'invalid_lines': bad, 'executable': False}, indent=2) + '\n')
            raise
        validate_linuxcnc_comments(text)
        executable_state(text, request)  # malformed comments/missing TC fail before file publication
        Path(out_path).write_text(text)
        return 'freecad:' + post_name
    finally:
        job.Operations.Group = original
        job.PostProcessorArgs, job.PostProcessorOutputFile = old_args, old_output
