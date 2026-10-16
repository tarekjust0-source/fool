"""Prompt composition: turns user settings into provider-ready text + params.

Every style control maps onto the request: either text in the prompt /
negative prompt, or a provider parameter (strength, style preset).  The UI
offers "Preview Prompt" so users can see exactly what is sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STYLES = {
    "REALISTIC": ("Realistic",
                  "physically based realistic materials, natural colour variation, believable surface detail",
                  "cartoon, cel shading, flat colours, painterly strokes",
                  "photographic"),
    "STYLIZED": ("Stylized",
                 "stylized game art, simplified readable shapes, strong colour design, clean material separation",
                 "photograph, photorealistic, noisy micro detail, film grain",
                 "digital-art"),
    "HAND_PAINTED": ("Hand-painted",
                     "hand-painted texture, visible soft brush strokes, painted highlights and shadows, "
                     "World of Warcraft / Overwatch inspired painterly style",
                     "photograph, photorealistic, photo texture, noise, film grain, harsh specular",
                     "fantasy-art"),
    "LOW_POLY": ("Low-poly game art",
                 "low-poly game asset texture, large flat colour areas, subtle gradients, minimal fine detail",
                 "photorealistic, intricate micro detail, noise, grain",
                 "low-poly"),
    "FANTASY": ("Fantasy",
                "high fantasy style, ornate details, rich saturated colours",
                "modern, sci-fi, photographic noise",
                "fantasy-art"),
    "SCIFI": ("Sci-fi",
              "science-fiction design, panel lines, technical markings, clean hard-surface materials",
              "medieval, organic clutter, painterly smudges",
              "digital-art"),
    "CARTOON": ("Cartoon",
                "cartoon style, bold flat colours, clean shapes, simple shading",
                "photograph, photorealistic, gritty detail, noise",
                "comic-book"),
    "PIXEL_ART": ("Pixel art",
                  "pixel art texture, limited colour palette, crisp pixel clusters, no anti-aliasing",
                  "photograph, smooth gradients, blur, high-resolution detail",
                  "pixel-art"),
}

MATERIAL_WORDS = {
    "AUTO": "",
    "METAL": "metal surfaces",
    "PAINTED_METAL": "painted metal",
    "WOOD": "wood with visible grain",
    "LEATHER": "leather",
    "STONE": "stone",
    "FABRIC": "woven fabric",
    "SKIN": "skin",
    "PLASTIC": "plastic",
    "CERAMIC": "ceramic",
}


def detail_words(v: float) -> str:
    if v < 0.2:
        return "very minimal detail, large clean shapes"
    if v < 0.45:
        return "moderate, restrained detail"
    if v < 0.75:
        return "rich surface detail"
    return "highly detailed, intricate surface detail"


def wear_words(v: float) -> str:
    if v < 0.1:
        return "pristine, brand new condition"
    if v < 0.35:
        return "lightly used, subtle wear on edges"
    if v < 0.7:
        return "noticeably worn, scratches and edge wear"
    return "heavily weathered and damaged, chipped, scratched, grime in crevices"


def hex_color(c) -> str:
    r, g, b = (max(0, min(255, int(round(x * 255)))) for x in c[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


@dataclass
class StyleSettings:
    style: str = "STYLIZED"
    material: str = "AUTO"
    detail: float = 0.5
    wear: float = 0.3
    strength: float = 0.7          # 0 = stick to input renders/reference, 1 = follow prompt freely
    use_palette: bool = False
    palette: list = field(default_factory=list)  # list of (r,g,b) 0..1 (sRGB)
    delight: bool = True           # request flat, unlit albedo
    art_direction: str = ""        # shared prompt applied to every object


@dataclass
class ComposedPrompt:
    prompt: str
    negative: str
    style_preset: str | None
    strength: float

    def as_dict(self) -> dict:
        return {"prompt": self.prompt, "negative": self.negative,
                "style_preset": self.style_preset, "strength": self.strength}


TASK_TEXT = {
    "first_view": (
        "This image is a render of a 3D model ({view} view) shown as untextured grey clay. "
        "Paint the surface of the model with this design: {subject}. "
        "Keep the exact silhouette, pose, framing and camera; do not add, remove or reshape geometry; "
        "do not change the background."
    ),
    "next_view": (
        "This image is a render of a partly textured 3D model ({view} view). "
        "Paint ONLY the grey clay areas (the masked region) so they continue the existing texture "
        "seamlessly: same design, colours, materials, scale of detail and art style. "
        "Design: {subject}. Keep the exact silhouette and camera; do not change the background."
    ),
    "refine": (
        "This image shows a textured 3D model. Edit ONLY the masked region: {subject}. "
        "Blend the edit seamlessly into the surrounding texture, matching lighting, scale and style. "
        "Do not change anything outside the masked region and keep the silhouette identical."
    ),
    "refine_uv": (
        "This image is part of a UV texture atlas of a 3D model. Repaint ONLY the masked region: {subject}. "
        "Keep the same flat texture-atlas presentation, blend seamlessly with surrounding pixels, "
        "do not add perspective or lighting."
    ),
    "tile": (
        "A seamless, tileable, flat top-down texture swatch (no perspective, no object, no border): {subject}."
    ),
}


def compose(user_prompt: str, user_negative: str, style: StyleSettings, *, purpose: str,
            view_name: str = "front", has_reference: bool = False, has_mask_image: bool = False,
            design_context: str = "") -> ComposedPrompt:
    if purpose not in TASK_TEXT:
        raise ValueError(f"Unknown purpose {purpose}")
    st = STYLES.get(style.style, STYLES["STYLIZED"])
    subject_parts = [user_prompt.strip() or "a coherent material design that suits the object"]
    if style.art_direction.strip():
        subject_parts.append(style.art_direction.strip())
    subject = "; ".join(subject_parts)
    lines = [TASK_TEXT[purpose].format(view=view_name, subject=subject)]
    if design_context.strip() and purpose in ("refine", "refine_uv"):
        lines.append(f"The object's overall design is: {design_context.strip()}. Keep the edit consistent with it.")
    lines.append(f"Art style: {st[1]}.")
    mat = MATERIAL_WORDS.get(style.material, "")
    if mat:
        lines.append(f"Primary material: {mat}.")
    lines.append(f"Detail: {detail_words(style.detail)}. Condition: {wear_words(style.wear)}.")
    if style.use_palette and style.palette:
        cols = ", ".join(hex_color(c) for c in style.palette)
        lines.append(f"Use this colour palette: {cols}.")
    if style.delight and purpose != "refine_uv":
        if style.style in ("HAND_PAINTED", "CARTOON", "LOW_POLY", "STYLIZED"):
            lines.append("Use soft, even lighting; painted shading is fine but avoid cast shadows "
                         "and strong directional lighting so the texture works from every angle.")
        else:
            lines.append("Render as flat albedo/diffuse colour under even, shadowless lighting: "
                         "no cast shadows, no specular highlights, no reflections.")
    if has_reference:
        lines.append("Use the additional reference image(s) as the guide for design, colours and materials; "
                     "do not copy its background or camera.")
    if has_mask_image:
        lines.append("A black-and-white mask image is attached: white marks the region to paint.")
    if style.strength < 0.35:
        lines.append("Stay very close to the provided images; make conservative changes.")
    elif style.strength > 0.8:
        lines.append("You may reinterpret details freely to match the description.")
    lines.append("No text, no logos, no watermarks.")
    neg = [n for n in (user_negative.strip(), st[2], "text, watermark, signature, blurry, extra objects") if n]
    return ComposedPrompt(" ".join(lines), ", ".join(neg), st[3], float(style.strength))
