# **The Quantitative Architecture of Tonal Fidelity: A Comprehensive Sensitometric Analysis of Emulsion Response and Paper Grading Systems**

The scientific foundation of photography and the graphic arts rests upon the discipline of sensitometry—the quantitative study of the response of light-sensitive materials to radiant energy and chemical processing. This field, which derives its name from the Latin roots for sensitivity and measurement, provides the empirical framework necessary to transform the subjective act of seeing into the objective process of reproduction.1 At its most fundamental level, sensitometry seeks to establish a predictable, mathematical relationship between the amount of light received by an emulsion and the resulting density of the image. The historical genesis of this study is credited to Ferdinand Hurter and Vero Charles Driffield, who, around 1876, began investigating the photochemical properties of early black-and-white emulsions.3 Their work moved photography beyond a craft based on trial and error into a rigorous science, establishing the "characteristic curve" as the primary tool for evaluating film and paper performance.3

## **Historical Foundations and the Hurter-Driffield Legacy**

The investigation into how the density of silver produced in an emulsion varies with the amount of light and the conditions of development began with the Photochemical Investigations of Hurter and Driffield, published in 1890\.3 Prior to their work, there was no standardized method for determining the "speed" or sensitivity of a photographic plate. Their introduction of the D-log E curve—alternatively known as the H\&D curve or the sensitometric curve—revolutionized the industry by plotting the developed density against the common logarithm of exposure.3 This logarithmic approach was dictated not only by the convenience of the scale but also by the physical relationship between density and the mass of developed silver per unit area.5  
Early attempts at measuring sensitivity were often unreliable due to spectral sensitivity issues and the fading intensity of light sources. The Warnerke Standard Sensitometer, commercialized in 1881, utilized a phosphorescent tablet and an opaque screen with 25 numbered squares, but its results were difficult to reproduce consistently.6 Hurter and Driffield addressed these inconsistencies by defining speed numbers that were inversely proportional to the exposure required to produce a specific photographic effect.6 This paved the way for more sophisticated systems such as the GOST scale in the Soviet Union and the ASA and DIN systems in the West, eventually converging into the modern ISO standards.6  
The technical depth of sensitometry was further expanded by Kenneth Mees and Loyd Jones at the Kodak Research Laboratories. Mees’ definitive work, *The Theory of the Photographic Process*, synthesized the physical nature of light with the chemical mechanisms of the latent image.8 Jones, meanwhile, focused on the psychophysical aspects of tone reproduction, determining how sensitometric curves could be manipulated to yield a "first-choice" print that satisfied human visual perception.9 These pioneers recognized that sensitometry is not an end in itself but a means to understand how an artist's visualization can be effectively captured in a negative and translated onto paper.2

## **Mathematical Principles of Optical Density and Exposure**

The characteristic curve is a Cartesian plot where the horizontal axis represents the input (exposure) and the vertical axis represents the output (density).13 In photographic terms, exposure $(E)$ is defined as the product of light intensity $(I)$ and the time of action $(t)$, typically expressed in lux-seconds.5 To manage the vast dynamic range of light encountered in the physical world, the common logarithm of exposure is used, which compresses the data and aligns it with the way the human eye perceives differences in brightness.4  
Density $(D)$ is the unit of image saturation and is defined as the logarithm of opacity $(O)$. Opacity is the reciprocal of transparency $(T)$, which is the ratio of transmitted light $(I\_t)$ to incident light $(I\_o)$.1 The mathematical relationship is expressed as:

$$D \= \\log\_{10}(O) \= \\log\_{10}\\left(\\frac{1}{T}\\right) \= \\log\_{10}\\left(\\frac{I\_o}{I\_t}\\right)$$  
A perfectly transparent area has a transmission of 1.0 and a density of 0.0, while a density of 1.0 indicates that only 10% of the light is transmitted.4 A density of 2.0 represents a transmission of 1% (an opacity of 100).3 In practical photography, a transmission densitometer is used for film, while a reflection densitometer is used for paper prints.1 The use of density is particularly advantageous because, in a multi-generational process, the end-to-end contrast ratio is determined by the anti-logarithm of the difference between the maximum and minimum densities $(D\_{max} \- D\_{min})$.3

| Transmission (%) | Opacity (O) | Density (D) | Visual Representation |
| :---- | :---- | :---- | :---- |
| 100% | 1.0 | 0.0 | Transparent / Pure White |
| 50% | 2.0 | 0.3 | Light Gray |
| 10% | 10.0 | 1.0 | Medium Gray |
| 1% | 100.0 | 2.0 | Dark Gray / Black |
| 0.1% | 1000.0 | 3.0 | Deep Black |

## **The Morphology of the Shadow Toe: Speed Criteria and Gradient Analysis**

The characteristic curve of most silver halide emulsions follows a distinct "S" shape, which can be divided into four functional regions: the toe, the straight-line portion, the shoulder, and the region of solarization.5 The "toe" is the lower portion of the curve where the density first begins to rise above the base-plus-fog level—the inherent density of the film support and the minimal silver developed in unexposed areas.1  
In the toe region, equal increments of log exposure do not produce equal increments of density. Instead, the gradient $(\\Delta D / \\Delta \\log E)$ increases gradually as exposure increases.1 This phenomenon leads to tonal compression in the shadows, meaning that subtle differences in scene brightness are not as effectively separated as they are in the midtones.1 Despite this compression, the upper portion of the toe is highly useful for recording shadow detail, and many professional exposure systems, such as the Zone System, intentionally place important dark subjects on the toe to balance speed and detail.1

### **The Evolution of Speed Criteria: From Threshold to Fractional Gradient**

The technical determination of film speed—the measure of an emulsion's sensitivity to light—has historically been centered on the toe of the curve.6 Because shadow detail is the most critical factor in producing a high-quality print, the speed point must be located at a point that guarantees a minimum useful gradient.19  
Several criteria have been employed throughout the history of sensitometry to define this speed point:

* **Threshold Criterion:** This is the point where the density is just perceptibly higher than the fog level. While simple, it does not account for the contrast required to separate shadow tones.6  
* **Inertia Criterion:** Developed by Hurter and Driffield, this point is found by extending the straight-line portion of the curve until it intersects the base-plus-fog line on the exposure axis. This method proved unreliable because the inertia point often shifts with development time.6  
* **Fixed Density Criterion:** Adopted by systems like DIN, this defines speed based on the exposure required to reach a specific density above fog, typically 0.10. While easy to measure, it does not correlate perfectly with the perceived quality of a print because it ignores the gradient.6  
* **Minimum Useful Gradient Criterion:** This criterion places the speed point where the gradient first reaches a specific value, such as 0.2. This more accurately reflects the film's ability to render detail.6  
* **Fractional Gradient Criterion:** Proposed by Loyd Jones, this is widely considered the most accurate predictor of shadow quality. It defines the speed point $(m)$ as the place where the gradient is 0.3 times the average gradient $(\\bar{G})$ of a 1.5 log-H range extending to the right of the point.6

### **The Delta-X Criterion and Modern ISO Standards**

The fractional gradient method, while superior in accuracy, was criticized for being difficult to calculate in a production environment.11 This led to the development of the Delta-X criterion, which allows for the simplicity of a fixed density measurement while retaining the benefits of the fractional gradient method.20  
Kodak researchers found that when film is developed to a specific contrast level—where the density at a point 1.3 log units to the right of the 0.1-above-fog point is exactly 0.80 units higher—the fractional gradient point consistently falls about one stop (0.296 log units) to the left of the fixed density point.11 This discovery allowed for the standardization of ISO speed, where the speed point is determined at a density of 0.1 above base-plus-fog, provided the development time is adjusted to meet the 0.80 density-difference requirement.11  
The mathematical relationship used to approximate fractional gradient speeds from fixed density points for various development levels is given by a parabolic equation:

$$\\Delta X \= 0.83 \- 0.86\\Delta D \+ 0.24\\Delta D^2$$  
In this equation, $\\Delta D$ represents the density difference over a 1.3 log exposure range, and $\\Delta X$ is the correction factor applied to the fixed density point to find the effective fractional gradient speed.11 This ensures that as development contrast increases or decreases, the effective speed rating remains tied to the preservation of shadow separation.11

## **The Dynamics of the Highlight Shoulder and Solarization**

Opposite the toe is the "shoulder," the region of overexposure where the rate of density increase begins to slow down as the available silver halide grains are exhausted.1 In this region, the curve bends away from the straight-line portion and eventually becomes parallel to the log exposure axis.5 This flattening indicates severe tonal compression, where highlights such as white clouds, bright reflections, or direct light sources begin to "block" or "burn out," losing all texture and detail.1  
The point of maximum density $(D\_{max})$ represents the highest degree of blackening of which the film is capable.14 In black-and-white negative film, highlights are represented by high density; in reversal (transparency) film or positive prints, highlights are represented by the light or clear parts of the image.13 Most changes in contrast caused by varying developer type or time are observed in the straight-line and shoulder regions, while the toe remains relatively constant.2

### **Solarization and Efficiency Variations**

In cases of extreme overexposure, the characteristic curve may enter the region of solarization, where a further increase in exposure actually results in a decrease in developed density.5 This phenomenon is an anomaly of the latent image formation process and is influenced by the character of the emulsion and the specific developer used.17  
Research by Blair and Leighton suggests that the formation of the latent image is a summation of effects from many quanta of light energy, each with varying efficiency—defined as the increase in density per unit of energy absorbed.25 This efficiency is highest at the start of the straight-line portion and decreases toward the shoulder.25 Furthermore, intermittent exposure (multiple short bursts) often produces different effects than a single continuous exposure of the same total energy, a discrepancy that is most pronounced near the toe and diminishes toward the shoulder.25

## **The Sensitometry of Positive Reflection Materials**

The sensitometric characteristics of photographic paper differ significantly from those of film because prints are viewed by reflected light rather than transmitted light.3 This fundamental difference imposes a physical limit on the maximum density achievable; while a negative might reach a density of 3.0 or higher, a paper print rarely exceeds a $D\_{max}$ of 2.1 to 2.4, as the blackest blacks are limited by the surface reflection of the paper base and gelatin.23  
The core objective in photographic printing is to match the density range $(DR)$ of the negative to the log exposure range $(LER)$ of the paper.15 The $DR$ of a negative is the difference between its highest (highlight) and lowest (shadow) densities. The $LER$ of the paper is the range of log exposures required to produce a full scale of tones on that specific paper.15

### **Log Exposure Range (LER) and International Standard ISO 6846**

To help photographers select the correct contrast of paper, manufacturers have historically used "paper grades" (e.g., Grade 0 through Grade 5).15 However, because these grades were not standardized across the industry, the ISO 6846 standard was developed to classify papers based on their "ISO Range" (ISOR).15  
The ISOR is calculated by determining the useful LER and multiplying it by 100\. The standard defines the useful LER as the log exposure difference between two specific points on the paper's curve:

1. **$ID\_{min}$ (Textural Highlight):** A density of 0.04 above the base-plus-fog $(b+f)$.23  
2. **$ID\_{max}$ (Textural Shadow):** A density equal to 90% of the paper’s maximum density $(D\_{max})$.23

| ISO Paper Grade | Log Exposure Range (LER) | ISO Range (R) | Qualitative Description |
| :---- | :---- | :---- | :---- |
| Grade 0 | 1.40 \- 1.70 | R150+ | Very Soft (High Latitude) |
| Grade 1 | 1.15 \- 1.40 | R120 \- R140 | Soft |
| Grade 2 | 0.95 \- 1.15 | R100 \- R110 | Normal / Standard |
| Grade 3 | 0.80 \- 0.95 | R85 \- R90 | Hard / Contrasty |
| Grade 4 | 0.65 \- 0.80 | R70 \- R75 | Very Hard |
| Grade 5 | 0.50 \- 0.65 | R55 \- R60 | Extra Hard |
| Grade 6 | 0.35 \- 0.50 | R40 \- R45 | Ultra Hard (Graphic Arts) |

The "Normal" standard for a negative is typically based on a density range of 1.05, which fits squarely in the middle of a Grade 2 paper's LER.28 If a negative is too flat (e.g., $DR \= 0.70$), it requires a "harder" paper (Grade 4\) to expand the tonal range. If a negative is too contrasty (e.g., $DR \= 1.30$), it requires a "softer" paper (Grade 1\) to compress the tones into the paper's limited scale.28

### **Variable Contrast Emulsions and Spectral Response**

Modern black-and-white printing is dominated by variable contrast (VC) papers, which utilize multiple emulsion layers to achieve different LER values on a single sheet.15 Typically, these papers consist of a high-contrast emulsion sensitive to blue light and a low-contrast emulsion sensitive to green light.18 By using magenta (blue-passing) or yellow (green-passing) filters, the photographer can blend the response of these layers to achieve half-grades or even more granular control over contrast.18  
The physics of variable contrast paper allows for "split-grade printing," where a photographer provides part of the exposure through a soft filter to establish highlight detail and part through a hard filter to set the deep blacks and enhance midtone separation.18 This technique circumvents some of the limitations of single-grade papers by allowing local control of the tonal curve during the printing process.18

## **Tone Reproduction Theory: The Jones Diagram and Psychophysical Correlation**

A major challenge in sensitometry is that the reproduction of a scene is a multi-step process, with each step possessing its own non-linear characteristic curve.26 Loyd Jones developed the "tone reproduction diagram," or Jones Diagram, to illustrate how these successive dependencies interact to produce the final image.9  
A standard Jones Diagram uses four quadrants (though Jones's original system used eleven) to map the flow of light 9:

* **Quadrant I (Final Quality):** Represents the relationship between the luminance of the original object and the reflectance seen by the viewer of the final print.9 This is the ultimate "tone reproduction curve" (TRC).  
* **Quadrant II (The Negative):** Plots the film's characteristic curve, showing how scene luminances (input) become negative densities (output).9  
* **Quadrant III (The Positive):** Plots the paper's characteristic curve, showing how light transmitted through the negative (input) becomes print densities (output).9  
* **Quadrant IV (The Transition):** Often a simple 45-degree line used as a geometric aid to project values back to the original axes, though it can also represent the enlarger's characteristics.9

The Jones Diagram is essential for understanding how a loss of contrast in one region can be compensated for elsewhere. For example, if a film has a very long, flat toe (shadow compression), it might be necessary to use a high-contrast paper that emphasizes low-value separation to prevent the final print from appearing "muddy".18

### **Flare and Non-Image Forming Light**

A critical factor accounted for in a complete tone analysis is "flare"—non-image-forming light that enters the camera or the enlarger lens.24 Flare prevents the optical image from being a tone-for-tone counterpart of the scene.24 In the camera, flare tends to "fill in" the shadows of the negative, reducing the effective contrast in the toe region.24 In the darkroom, enlarger flare can veil the highlights of the print. Sensitometric calculations for a "Normal" negative often subtract a standard flare factor (e.g., 0.40 log units) from the scene's luminance range to arrive at the aim density range.28

### **Visual Adaptation and the Perception of Photographic Quality**

Tone reproduction is not merely an objective match of densities; it is a psychophysical phenomenon governed by how the eye perceives tones under different illumination levels.33 Research indicates that as the overall illuminance of a room decreases, the human eye perceives "white" as less bright, but "black" remains relatively constant in its perceived depth—a phenomenon known as brightness constancy.33  
Furthermore, the "surround" of a photograph significantly alters the perception of its internal contrast. A white matte border makes a gray scale appear darker and increases the apparent contrast of the highlights while decreasing the perceived separation in the shadows.33 This suggests that the "correct" negative and print density scales are not absolute but are dependent on the final viewing conditions and the artist's intent to reproduce subjective "brightness differences" rather than objective luminances.26

## **Practical Application: The Zone System and Emulsion Design**

The principles of sensitometry found their most enduring application in the Zone System, developed by Ansel Adams and Fred Archer.12 The Zone System provides a disciplined approach to "visualization," where scene luminances are measured and assigned to specific "Zones" (numbered 0 through X) on the characteristic curve.12

* **Zones II and III (The Shadow Range):** These are placed on the toe of the curve. Consistent with the fractional gradient criterion, these zones must have enough density and contrast to separate dark textures.18  
* **Zone V (Middle Gray):** The pivot point of the system, corresponding to 18% reflectance.12  
* **Zones VII and VIII (The Highlight Range):** These fall near the shoulder. The photographer uses "compressed" or "expanded" development to ensure these highlights have textural detail without reaching the flat part of the shoulder.18

Adams’ work demonstrated that by manipulating the negative's slope (gamma) through development time, a photographer could intentionally fit a wide-dynamic-range scene (High Dynamic Range or HDR) into the limited density range of a standard print (Low Dynamic Range or LDR).12 This spatial and tonal processing is the analog precursor to modern digital image processing and dodging and burning techniques.12

## **Halftone Printing and Dot Gain**

In the context of the graphic arts and commercial printing, sensitometry extends into the study of structured images, such as halftones.26 A halftone negative represents shades of gray through variable-sized dots of constant density.34 The sensitometry of this process must account for "dot gain"—the physical and optical spreading of ink as it hits the paper.26  
If a digital image specifies a 50% gray, the printer must apply less than 50% ink coverage to compensate for the bleed that will occur as the ink absorbs into the substrate.26 A tone reproduction curve (TRC) is applied to the electronic image to ensure the final print's reflectance matches the intended luminance.26 This demonstrates that the same principles of the H\&D curve—balancing input $(\\log E$ or dot percentage) against output (reflection density)—apply across all forms of visual reproduction.

## **Specific Developer Chemistry and pH Profiles**

The shape of the characteristic curve is not solely a property of the emulsion but is a result of the interaction between the emulsion and the developer chemistry.2 Research by Mees and James highlighted that each developing agent has a specific pH profile and an ionization constant that determines its effectiveness.36  
Different developers can provide either direct development (reducing the silver halide grains directly) or solution physical development (where silver is dissolved and then redeposited).36 For instance, fine-grain developers often utilize solution physical development to produce smaller silver clumps, which can subtly soften the toe of the curve and require a slight increase in exposure to maintain shadow detail.36 High-contrast developers, often used in radiography or graphic arts, produce a very steep straight-line portion with a sharp toe and shoulder, maximizing the separation of binary (black/white) tones.4

| Developer Type | Characteristic Effect on Curve | Application |
| :---- | :---- | :---- |
| High-Contrast (e.g., D-19) | Steep straight-line, sharp shoulder | Radiography, Technical, Graphics 4 |
| General Purpose (e.g., D-76) | Moderate gamma, long toe | Professional and Amateur Photography 11 |
| Fine Grain (e.g., DK-20) | Lower gamma, softer toe | Small format negatives requiring enlargement 36 |
| Compensating | Compressed shoulder, expanded toe | High-dynamic-range scenes 12 |

## **Technical Synthesis and the Future of Sensitometry**

The transition to digital imaging has not rendered sensitometry obsolete; rather, it has transformed its application. Digital sensors have a linear response to light, which differs from the "S" curve of film. To make digital images look "photographic," engineers apply a virtual characteristic curve—often referred to as a "gamma curve" or "Log curve"—during the analog-to-digital conversion or in post-production.26  
Digital log curves (e.g., S-Log2, Log C) are designed to mimic the highlight shoulder of film, allowing for more stops of dynamic range to be recorded without clipping.35 The "expose to the right" (ETTR) technique in digital photography is essentially a sensitometric strategy to move the shadows out of the noise floor (the digital "toe") while carefully monitoring the histogram to prevent the "shoulder" (digital clipping) from being reached.35  
In conclusion, sensitometry provides the rigorous language through which the complexities of light capture and reproduction are managed. Whether one is measuring the fractional gradient of a silver halide negative, determining the ISO range of a fiber-based paper, or calibrating the dot gain of a CMYK press, the fundamental goal remains the same: the mastery of the relationship between exposure and density.2 The legacy of Hurter, Driffield, and Jones continues to guide the evolution of visual media, ensuring that the art of photography remains firmly rooted in the science of light.3

#### **Cytowane prace**

1. Practical Densitometry E\_59 \- Conrad Hoffman, otwierano: stycznia 1, 2026, [https://conradhoffman.com/papers\_lib/Practical%20Densitometry%20E\_59.pdf](https://conradhoffman.com/papers_lib/Practical%20Densitometry%20E_59.pdf)  
2. Basic Photographic Sensitometry Workbook | Kodak, otwierano: stycznia 1, 2026, [https://www.kodak.com/content/products-brochures/Film/Basic-Photographic-Sensitometry-Workbook.pdf](https://www.kodak.com/content/products-brochures/Film/Basic-Photographic-Sensitometry-Workbook.pdf)  
3. Sensitometry \- Wikipedia, otwierano: stycznia 1, 2026, [https://en.wikipedia.org/wiki/Sensitometry](https://en.wikipedia.org/wiki/Sensitometry)  
4. C-Curve and Senitometry | PDF | Exposure (Photography) | Logarithm \- Scribd, otwierano: stycznia 1, 2026, [https://www.scribd.com/presentation/691286223/C-curve-and-Senitometry](https://www.scribd.com/presentation/691286223/C-curve-and-Senitometry)  
5. 1960, Photographic Theory by James and Higgins, Chapter 1 \- UCLA Astronomy, otwierano: stycznia 1, 2026, [https://astro.ucla.edu/\~ulrich/MW\_SPADP/1960James\_Higgins.pdf](https://astro.ucla.edu/~ulrich/MW_SPADP/1960James_Higgins.pdf)  
6. Film speed \- Wikipedia, otwierano: stycznia 1, 2026, [https://en.wikipedia.org/wiki/Film\_speed](https://en.wikipedia.org/wiki/Film_speed)  
7. Film speed \- Wikiwand, otwierano: stycznia 1, 2026, [https://www.wikiwand.com/en/articles/Film\_speed](https://www.wikiwand.com/en/articles/Film_speed)  
8. Kodak Color Densitometer \- phsc.ca, otwierano: stycznia 1, 2026, [https://phsc.ca/repair/Kodak/kodak\_densitometer\_1.pdf](https://phsc.ca/repair/Kodak/kodak_densitometer_1.pdf)  
9. Jones diagram \- Wikipedia, otwierano: stycznia 1, 2026, [https://en.wikipedia.org/wiki/Jones\_diagram](https://en.wikipedia.org/wiki/Jones_diagram)  
10. Zone Placement | Page 3 \- Photrio.com Photography Forums, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/threads/zone-placement.80739/page-3](https://www.photrio.com/forum/threads/zone-placement.80739/page-3)  
11. Simple Methods for Approximating the Fractional Gradient Speeds ..., otwierano: stycznia 1, 2026, [https://opg.optica.org/abstract.cfm?uri=josa-46-5-324](https://opg.optica.org/abstract.cfm?uri=josa-46-5-324)  
12. The Ansel Adams Zone System: HDR capture and range compression by chemical processing, otwierano: stycznia 1, 2026, [https://www.retinex2.net/Publications/ewExternalFiles/10EI%207527-28%20ADAMS.pdf](https://www.retinex2.net/Publications/ewExternalFiles/10EI%207527-28%20ADAMS.pdf)  
13. BASIC SENSITOMETRY AND CHARACTERISTICS OF FILM \- Kodak, otwierano: stycznia 1, 2026, [https://www.kodak.com/uploadedfiles/motion/US\_plugins\_acrobat\_en\_motion\_newsletters\_filmEss\_06\_Characteristics\_of\_Film.pdf](https://www.kodak.com/uploadedfiles/motion/US_plugins_acrobat_en_motion_newsletters_filmEss_06_Characteristics_of_Film.pdf)  
14. Film curve plotting and fitting | Page 15 \- Photrio.com Photography Forums, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/threads/film-curve-plotting-and-fitting.63864/page-15](https://www.photrio.com/forum/threads/film-curve-plotting-and-fitting.63864/page-15)  
15. ISO 6846:1992 \- iTeh Standards, otwierano: stycznia 1, 2026, [https://cdn.standards.iteh.ai/samples/13355/5b11c8fb17f843aab03de56691b4b353/ISO-6846-1992.pdf](https://cdn.standards.iteh.ai/samples/13355/5b11c8fb17f843aab03de56691b4b353/ISO-6846-1992.pdf)  
16. A film's senistometric density curve and determining ISO \- Filmwasters.com, otwierano: stycznia 1, 2026, [https://filmwasters.com/forum/index.php?topic=7608.0](https://filmwasters.com/forum/index.php?topic=7608.0)  
17. Title of paper, otwierano: stycznia 1, 2026, [https://isprs.org/proceedings/XXXV/congress/comm1/papers/15.pdf](https://isprs.org/proceedings/XXXV/congress/comm1/papers/15.pdf)  
18. Power of Process Exposure & Negative Design \- Steve Sherman, otwierano: stycznia 1, 2026, [https://www.powerofprocesstips.com/power-of-process-exposure-negative-design/](https://www.powerofprocesstips.com/power-of-process-exposure-negative-design/)  
19. Methodology and Curve Interpretation | Page 6 | Photrio.com Photography Forums, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/threads/methodology-and-curve-interpretation.189935/page-6](https://www.photrio.com/forum/threads/methodology-and-curve-interpretation.189935/page-6)  
20. Adventures in film characteristic analysis | Page 6 | Photrio.com Photography Forums, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/threads/adventures-in-film-characteristic-analysis.189526/page-6](https://www.photrio.com/forum/threads/adventures-in-film-characteristic-analysis.189526/page-6)  
21. Sensitometry \- US2380244A \- Google Patents, otwierano: stycznia 1, 2026, [https://patents.google.com/patent/US2380244A/en](https://patents.google.com/patent/US2380244A/en)  
22. Box ISO rate and Real ISO | Page 4 | Photrio.com Photography Forums, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/threads/box-iso-rate-and-real-iso.113575/page-4](https://www.photrio.com/forum/threads/box-iso-rate-and-real-iso.113575/page-4)  
23. Measuring Paper Contrast, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/attachments/measurepapercontrasted2-pdf.92471/](https://www.photrio.com/forum/attachments/measurepapercontrasted2-pdf.92471/)  
24. A Novel Graphical System for Representing Tone Reproduction Data, otwierano: stycznia 1, 2026, [https://opg.optica.org/abstract.cfm?uri=josa-34-10-597](https://opg.optica.org/abstract.cfm?uri=josa-34-10-597)  
25. The Intermittency Effect in Photographic Exposure \- Optica Publishing Group, otwierano: stycznia 1, 2026, [https://opg.optica.org/fulltext.cfm?uri=josa-23-10-353](https://opg.optica.org/fulltext.cfm?uri=josa-23-10-353)  
26. Tone reproduction \- Wikipedia, otwierano: stycznia 1, 2026, [https://en.wikipedia.org/wiki/Tone\_reproduction](https://en.wikipedia.org/wiki/Tone_reproduction)  
27. The Control of Photographic Printing by Measured Characteristics of ..., otwierano: stycznia 1, 2026, [https://opg.optica.org/fulltext.cfm?uri=josa-32-10-558](https://opg.optica.org/fulltext.cfm?uri=josa-32-10-558)  
28. What is Normal \- Photrio.com Photography Forums, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/attachments/what-is-normal-pdf.18213/](https://www.photrio.com/forum/attachments/what-is-normal-pdf.18213/)  
29. Tonal Range to Paper Grade Relationship Info? \- Photrio.com Photography Forums, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/threads/tonal-range-to-paper-grade-relationship-info.19136/](https://www.photrio.com/forum/threads/tonal-range-to-paper-grade-relationship-info.19136/)  
30. Paper Grades | Roger and Frances, otwierano: stycznia 1, 2026, [https://www.rogerandfrances.com/paper-grades/](https://www.rogerandfrances.com/paper-grades/)  
31. BS ISO 6846:1992 PDF \- PDF Standards Store \- livewell, otwierano: stycznia 1, 2026, [https://livewell.ae/customer\_feedback\_API/pdf.php?u=/product/publishers/bsi/bs-iso-68461992/](https://livewell.ae/customer_feedback_API/pdf.php?u=/product/publishers/bsi/bs-iso-68461992/)  
32. GSO ISO 6846:2015 \- Standards Store \- Ministry of Commerce and Industry, otwierano: stycznia 1, 2026, [https://dgsm.gso.org.sa/store/standards/GSO:691562?lang=en](https://dgsm.gso.org.sa/store/standards/GSO:691562?lang=en)  
33. Subjective Considerations | Photrio.com Photography Forums, otwierano: stycznia 1, 2026, [https://www.photrio.com/forum/threads/subjective-considerations.101979/](https://www.photrio.com/forum/threads/subjective-considerations.101979/)  
34. Tone Reproduction in the “Halftone” Photo-Engraving Process \- Optica Publishing Group, otwierano: stycznia 1, 2026, [https://opg.optica.org/abstract.cfm?uri=josa-13-5-537](https://opg.optica.org/abstract.cfm?uri=josa-13-5-537)  
35. How to expose LOG footage correctly \- Daniel Haggett, otwierano: stycznia 1, 2026, [https://www.danielhaggett.com/blog/262-how-to-expose-log-footage-correctly](https://www.danielhaggett.com/blog/262-how-to-expose-log-footage-correctly)  
36. Basic Library \- Unblinking Eye, otwierano: stycznia 1, 2026, [https://unblinkingeye.com/Articles/BLPE/blpe.html](https://unblinkingeye.com/Articles/BLPE/blpe.html)