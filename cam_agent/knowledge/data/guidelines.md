# Machining and DFM guidelines (text corpus for RAG retrieval)

## Process ordering: general rules (standard CAPP practice, GOST 3.1109 route principles)
Establish the datum first: face the top surface (and the clamping faces) before any feature that references it. Then
remove the largest stock: rough every feature of the setup before finishing any of them (stages separated), so that
residual stress, heat and clamping deformation are released before the final cuts. Holes are made after the faces and
floors they start from have been milled; the most accurate surfaces (reamed and bored holes, ground faces) come last.
Within a phase, group operations by tool so that each tool is loaded once, and order the features of one tool along a
short path. Deburring follows all material removal (ISO 13715); cleaning, inspection and coating follow off the machine,
the coating after the dimensional inspection.

## Hole making sequence
Spot drill (90 deg) -> drill -> chamfer or countersink the entry -> ream or bore -> tap or thread mill. The entry chamfer
is made before tapping and reaming so that no burr is pushed into the thread or bore. Carbide drills with a 140 deg point
are self-centring on flat surfaces, but a spot drill is still used when position tolerance is tight, on inclined or rough
surfaces and before HSS drills. A spot drill must not make a cone larger than the hole unless a chamfer is modelled.

## Heat treatment in the route
Hardening or quench-and-temper is placed between roughing and finishing. Before it: roughing with finishing stock, spot
drilling, drilling of fixing holes, counterbores, countersinks and tapped threads (a tap cannot cut steel hardened above
about 45 HRC). After it: finishing passes, reaming and boring of precision holes, grinding of toleranced faces, and thread
milling of threads that were not tapped. Grinding never precedes heat treatment, and no heat treatment follows a coating.

## Accuracy of machining methods (economic accuracy)
Rough milling IT12, Ra 6.3 um; finish milling of contours, pockets and slots IT7-IT8, Ra 1.6 um; face milling IT8, Ra 1.6 um;
drilling IT11, Ra 6.3 um; countersinking and counterboring IT10, Ra 3.2 um; reaming IT7, Ra 0.8 um; fine boring IT6,
Ra 0.8 um; tapping 6H; thread milling 6H, Ra 1.6 um; surface grinding IT6, Ra 0.4 um. A requirement tighter than every
planned method needs an extra finishing step: reaming or boring for holes, grinding for faces.

## Holes: drilling practice
Twist drilling covers 1-40 mm. Peck (G83) when depth exceeds about 3xD with HSS drills or 5xD with carbide drills without
through-coolant; peck depth 0.5-1xD in steel and stainless, 1.5-2xD in aluminium and cast iron. For through holes add the
drill point (0.3xD for 118 deg, 0.18xD for 140 deg) plus about 1 mm of break-through. A drilled hole holds IT10-IT12 and
Ra 3.2-6.3 um. Feed per revolution scales with the diameter (about 0.02xD in steel, 0.03xD in aluminium, 0.015xD in stainless).

## Holes: reaming and boring
Reaming follows drilling for IT7-IT8 and Ra 0.8-1.6 um. Diameter stock for the reamer (VDI 3210): 0.1-0.15 mm up to 5 mm,
0.2 mm up to 10 mm, 0.25 mm up to 20 mm, 0.3 mm up to 50 mm. Reamer speed is about one third to one half of the drilling
speed and the feed two to three times the drill feed; the cycle feeds in and out (G85). Boring is used for IT6-IT7, for
large diameters, or when the hole position must be corrected; leave 0.5-1 mm on diameter.

## Threads: tapping and thread milling
Tap drill = nominal diameter minus pitch (ISO 2306, about 75 % thread height): M6 -> 5.0 mm, M8 -> 6.8 mm, M10 -> 8.5 mm,
1/4-20 UNC -> 5.1 mm (#7). Tapping must be a rigid tapping cycle (G84, or G33.1 on LinuxCNC) with feed = pitch x spindle
speed and spindle reversal at the bottom; a plain drilling cycle (G81) retracts at rapid without reversing and breaks the
tap. Spiral-flute taps for blind holes, spiral-point taps for through holes. In stainless steel, titanium and nickel alloys
use 55-65 % thread and prefer thread milling from M10; in hardened steel threads are thread-milled only. Blind tapped
holes need the drill 2-3 pitches deeper than the full thread (DIN 76 run-out).

## Pockets and slots
Internal corners need a radius larger than the tool radius (about 1.2x) so that the cutter does not wrap around the corner.
Rough pockets with adaptive (trochoidal) paths: radial engagement 5-15 % of D and axial depth up to 2xD in aluminium,
1-1.5xD in steel; conventional offset clearing uses 30-50 % stepover and 0.5xD depth in steel. Leave 0.2-0.5 mm on walls
and 0.1-0.2 mm on floors for finishing. Avoid full slotting (ae = D) in steel and stainless; when it cannot be avoided cut
at most 0.3xD (steel) or 0.2xD (stainless) deep per pass. A pocket depth above 4x its narrowest width needs long-reach tools
and reduced ap and ae.

## Nested features
A deeper pocket, slot or hole that opens from the floor of another pocket or slot is machined after that container has
been roughed, and it starts at the container floor, not at the top of the part. The container is cleared over its whole
footprint down to its own floor first.

## Facing and datum establishment
Face the top surface first. Face mill diameter about 1.3-1.6 times the cut width, stepover 60-75 % of the cutter diameter,
0.5-3 mm per pass. Enter the work off-centre (climb) to avoid chip thinning at the entry.

## Chamfers and deburring
A modelled 45 deg chamfer is milled with a 90 deg chamfer mill whose flank lies on the face; other angles are finished with
a ball endmill in fine steps. All chamfers that use the same tool are machined in one operation. Edges without a specified
condition are broken 0.1-0.3 mm (ISO 13715 edge symbol, -0.3 typical).

## Cutting data model
Spindle speed n = Vc x 1000 / (pi x D); table feed vf = fz x z x n; drilling vf = f x n; tapping vf = pitch x n. The chip
load of an endmill grows with its diameter (fz about 0.006xD in steel, 0.010xD in aluminium, 0.004xD in titanium and
nickel alloys). At radial engagement below 50 % the chip is thinner than fz (radial chip thinning): fz is raised by
D / (2 sqrt(ae (D - ae))), limited to 2.5x. Spindle power Pc = ap ae vf kc / (60e6 eta) with kc = kc1.1 hm^-mc (Kienzle);
when it exceeds the machine, axial depth is reduced first. Long-reach tools (stick-out above 5xD) run at about 80 % speed,
70 % feed and half the axial depth.

## Materials: ISO 513 groups
P steels (unalloyed, low-alloy, tool steel unhardened), M stainless steels (austenitic, duplex), K cast irons (grey, ductile),
N non-ferrous (aluminium, copper alloys, plastics), S heat-resistant superalloys and titanium, H hardened steels (> 45 HRC)
and chilled cast iron. Coatings: TiAlN/AlTiN for P, M, K, S, H; uncoated polished, ZrN or DLC for aluminium (TiAlN causes
built-up edge); AlCrN for titanium and nickel alloys.

## Materials: aluminium alloys (EN AW-6061, 7075, 5083)
Carbide 300-600 m/min, HSS 90-150 m/min. Two or three flutes for chip room. Flood or MQL against chip welding. Minimum wall
1 mm, minimum internal radius 0.5 mm.

## Materials: stainless steels (1.4301 / 304, 1.4404 / 316L)
Work hardening: chip load above 0.03 mm per tooth, never rub or dwell, Vc 80-150 m/min with coated carbide, high-pressure
flood. Parabolic-flute drills, peck above 3xD. Thread milling preferred for M10 and larger.

## Materials: steels (S235 / 1018, C45 / 1045, 42CrMo4 / 4140)
Carbide 120-260 m/min depending on hardness, HSS 20-35 m/min. 42CrMo4+QT (28-32 HRC) needs a rigid set-up.

## Materials: cast iron (EN-GJL-250, EN-GJS-500-7)
Short abrasive chips: dry machining or air blast with dust extraction for grey iron; ductile iron benefits from coolant.

## Materials: titanium and nickel alloys (Ti-6Al-4V, Inconel 718)
Low thermal conductivity: low speed (Ti 40-80, Inconel 20-40 m/min with carbide), light radial engagement, constant chip
load, high-pressure coolant, sharp AlCrN/AlTiN tools. Thread milling instead of tapping.

## Materials: hardened steels (48-55 HRC)
Hard milling with small radial engagement (2-5 %), AlTiN/TiSiN carbide, air blast or MQL. No HSS tools, no tapping.

## Materials: plastics (POM-C, PEEK)
Sharp polished one or two flute tools, air blast, light clamping. Anneal PEEK before precision work.

## Tolerances and surface finish
General tolerances ISO 2768-1: class m allows +-0.1 mm up to 6 mm, +-0.2 mm for 6-30 mm, +-0.3 mm for 30-120 mm,
+-0.5 mm for 120-400 mm; class f is about half of m. Fits follow ISO 286 (H7 holes by reaming or boring). Achievable
roughness (ISO 21920): milling Ra 0.8-3.2 um, drilling Ra 3.2-6.3 um, reaming Ra 0.8-1.6 um, boring Ra 0.4-1.6 um,
grinding Ra 0.1-0.8 um. Process capability: drilling IT10-IT12, milling IT8-IT11, reaming IT7-IT8, boring IT6-IT7,
grinding IT5-IT6.

## Fixturing and stock
Sawn plate or bar: 2-3 mm allowance per side; castings 3-5 mm. A vice needs 5-10 mm of grip; features on the clamped faces
or the bottom need a second setup (flip, with soft jaws or a fixture). Work offsets are set on the stock (top and a corner)
or on a machined datum in later setups.

## Coolant and chip control
Flood for steel, stainless, titanium, nickel alloys and aluminium (or MQL). Grey cast iron dry or air blast. Brass and
plastics dry, air blast or mist. Deep pockets: air blast and periodic retracts for chip evacuation.
