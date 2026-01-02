# DarkroomPy: Core Image Processing Logic

DarkroomPy is built on a "True Darkroom Model" that replicates the physical workflow of analog photography. Instead of traditional digital sliders, the algorithms follow the path of light through a negative and a lens onto paper.

## 1. The True Darkroom Model
The processing pipeline is structured as follows:
**Filtered Light → Negative → Scanner Gain (Grade) → Inversion → Positive.**

By working in the "Negative Domain" (before inversion), we can apply adjustments that mimic physical processes like dichroic filtration and paper grade selection. This approach ensures that color and tonality are handled with the same mathematical integrity as light hitting a sensor or film.

## 2. White Balance (Dichroic Filtration)
In a traditional color darkroom, white balance is achieved using CMY (Cyan, Magenta, Yellow) filters in the enlarger head. 

### Logic & Implementation
- **Mathematical Basis**: We treat the RAW data as a physical negative. The first step is to neutralize the "orange mask" of the film base.
- **Algorithm**: We apply linear multipliers to the Red, Green, and Blue channels.
- **Rationale**: Unlike digital "Temperature/Tint" sliders which often use complex color space rotations, our filtration happens at the earliest possible stage. This preserves the chromaticity ratios of the scene, ensuring that as you adjust "exposure" later, the colors don't "drift"—a principle often emphasized in the cinematography research of [Steve Yedlin, ASC](https://www.yedlin.net/DisplayPrepDemo/index.html).

## 3. Density & Grade (Tonality)
In the darkroom, "Density" refers to the amount of silver/dye developed on the paper (brightness), while "Grade" refers to the contrast of the photographic paper. Harder grades (Grade 5) have a steeper response, while softer grades (Grade 0) compress more dynamic range.

### Logic & Implementation
- **Mathematical Basis**: We model this using a Photometric Characteristic Curve.
- **Algorithm**: 
  - The **Density** slider controls the curve pivot (Exposure Shift). 
    - `Density 1.0` is considered normal.
    - `Density > 1.0` darkens the print.
  - The **Grade** slider controls the curve slope (Contrast).
    - `Grade 2.0` is considered normal.
    - `Grade > 2.0` increases contrast (harder paper).
    - `Grade < 2.0` decreases contrast (softer paper).
- **Photographic Rationale**: This mimics matching the density range of the negative to the log exposure range of the paper ([ISO 6846:1992](https://cdn.standards.iteh.ai/samples/13355/5b11c8fb17f843aab03de56691b4b353/ISO-6846-1992.pdf)).

## 4. Tonality Recovery: Shoulder & Toe
Film and paper don't have a hard "clip" like digital sensors; they have a graceful roll-off at the extremes. We implement this using sensitometric curve logic during the gain stage.

### Shadow Toe (Negative Recovery)
- **Problem**: In a digital inversion, the deepest parts of the negative (the mask) become the blackest shadows. Without a "toe," these shadows clip abruptly.
- **Implementation**: We use a rational recovery function (`0.5 + (v - 0.5) / (1 + k(v - 0.5))`) to compress the negative's highest values.
- **Result**: This "opens up" the shadows in the final print, preventing "ink-black" blocks and retaining texture in dark areas, mimicking the [fractional gradient criteria](https://opg.optica.org/abstract.cfm?uri=josa-46-5-324) used for determining film speed.

### Highlight Shoulder (Negative Compression)
- **Problem**: Peak highlights in the negative (the densest areas) can easily exceed the range of the paper, leading to "blown-out" whites.
- **Implementation**: We apply a logarithmic compression to the lowest intensities of the negative.
- **Result**: This creates a "shoulder" in the positive, rolling off highlights smoothly into peak white. This mimics the natural compression of photographic paper and is based on [John Hable’s Filmic Tone Mapping](http://filmicworlds.com/blog/filmic-tonemapping-operators/) research.

## 5. Scientific Foundations & References
The algorithms in DarkroomPy are grounded in the intersection of color science and sensitometry:
- **Hurter & Driffield**: Established the relationship between exposure and density (H&D Curves). [Sensitometry - Wikipedia](https://en.wikipedia.org/wiki/Sensitometry).
- **Loyd Jones**: Developed the [Jones Diagram](https://en.wikipedia.org/wiki/Jones_diagram) for tone reproduction, which we use to map negative densities to print reflectances.
- **Steve Yedlin, ASC**: Modern cinematography research on [Display Prep and Data Modeling](https://www.yedlin.net/DisplayPrepDemo/index.html).
- **Kodak Research Labs**: Foundational work on [Photographic Sensitometry](https://www.kodak.com/content/products-brochures/Film/Basic-Photographic-Sensitometry-Workbook.pdf) and the [Theory of the Photographic Process](https://unblinkingeye.com/Articles/BLPE/blpe.html).
- **Ansel Adams**: The [Zone System](https://www.retinex2.net/Publications/ewExternalFiles/10EI%207527-28%20ADAMS.pdf) for HDR capture and chemical range compression.

By combining these scientific approaches, DarkroomPy offers a toolset that feels familiar to an analog printer but provides the precision and flexibility of a modern digital workflow.
