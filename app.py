"""MyFont - Local Font Generator

Single-page Streamlit application with tabs.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import sys

sys.path.insert(0, '.')

st.set_page_config(
    page_title="MyFont - Font Generator",
    page_icon="🔤",
    layout="wide",
)

from ui.state import (
    init_state, get_state, set_uploaded_image, set_processed_image,
    set_glyphs, set_mapping, remove_mapping, GlyphData,
    merge_selected_glyphs
)
from myfont.image_processing.loader import load_image, validate_image, get_image_info
from myfont.image_processing.preprocessor import preprocess_image, auto_detect_invert
from myfont.image_processing.segmenter import segment_glyphs, merge_nearby_contours
from myfont.character_mapping.charset import UPPERCASE, LOWERCASE, DIGITS
from myfont.vectorization.potrace_wrapper import check_potrace_installed
from myfont.font_generation.font_builder import build_font
from myfont.font_generation.exporter import get_all_formats_bytes, get_font_info, validate_font, get_cbdt_bytes
from myfont.character_mapping.charset import get_glyph_name
from myfont.preview.renderer import (
    render_text_preview, create_glyph_grid,
    render_color_font_preview, create_color_glyph_grid, is_color_font,
)
from myfont.ai_generation import (
    STYLE_PRESETS, CHARACTER_SETS, build_prompt,
    GeminiClient, GeminiError, GEMINI_AVAILABLE,
)
from myfont.ai_generation.prompts import CHARACTER_SET_LABELS, get_style_description
from myfont.credits import get_credits, create_credit_code, use_credit
from myfont.payments import verify_payment

def remove_background(image: np.ndarray, threshold: int = 240) -> np.ndarray:
    """Remove near-white background from image by making it transparent.

    Args:
        image: BGR image from OpenCV.
        threshold: Pixels with all channels above this are considered background.

    Returns:
        BGRA image with transparent background.
    """
    # Convert to BGRA (add alpha channel)
    if len(image.shape) == 2:
        # Grayscale - convert to BGRA
        bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        # BGR - convert to BGRA
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] == 4:
        # Already BGRA
        bgra = image.copy()
    else:
        return image

    # Find near-white pixels (background)
    # A pixel is background if all color channels are above threshold
    b, g, r, a = cv2.split(bgra)
    background_mask = (b > threshold) & (g > threshold) & (r > threshold)

    # Set alpha to 0 for background pixels
    a[background_mask] = 0

    # Recombine
    bgra = cv2.merge([b, g, r, a])

    return bgra


def normalize_background(color_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Use binary mask to set background pixels to pure white in color image.

    Args:
        color_image: BGR color image from original upload.
        binary_mask: Binary image where white (255) = foreground, black (0) = background.

    Returns:
        Color image with background pixels set to pure white (255,255,255).
    """
    result = color_image.copy()

    # Resize mask if dimensions don't match (due to padding differences)
    if binary_mask.shape[:2] != color_image.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (color_image.shape[1], color_image.shape[0]))

    # Background is where mask is black (0)
    background = binary_mask < 128

    # Set background pixels to pure white
    result[background] = [255, 255, 255]

    return result


# Initialize state
init_state()

# Handle payment callback - generate and show credit code
params = st.query_params
if params.get("session_id"):
    if verify_payment(params.get("session_id")):
        # Generate new credit code
        new_code = create_credit_code(3)
        st.session_state["new_credit_code"] = new_code
    st.query_params.clear()

# Show the new code prominently if just purchased
if st.session_state.get("new_credit_code"):
    st.success("Payment successful!")
    st.warning(f"Your credit code: **{st.session_state['new_credit_code']}**")
    st.info("Save this code! You'll need it to use your credits.")
    if st.button("I've saved my code"):
        del st.session_state["new_credit_code"]
        st.rerun()

st.title("🔤 MyFont - Local Font Generator")
st.markdown("Convert glyph images into usable font files (TTF, WOFF, WOFF2)")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "✂️ Segment", "🔤 Map", "⬇️ Generate"])

# ============ TAB 1: UPLOAD ============
with tab1:
    state = get_state()  # Fresh state for this tab
    st.header("Image Source")

    st.markdown("""
    Provide a glyph sheet containing all your characters.
    You can upload an existing image or generate one with AI.
    """)

    # Source selection
    source_mode = st.radio(
        "How would you like to create your glyph sheet?",
        options=["upload", "generate"],
        format_func=lambda x: "Upload Image" if x == "upload" else "Generate with AI",
        horizontal=True,
        key="source_mode",
    )

    st.divider()

    if source_mode == "upload":
        # ---- UPLOAD PATH (existing flow) ----
        st.subheader("Upload Glyph Image")
        st.markdown("**Tips:** Use high contrast (black on white), keep glyphs well-separated.")

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
            key="uploader"
        )

        # Handle new file upload
        if uploaded_file is not None:
            try:
                # Only load if this is a new file
                if state.uploaded_filename != uploaded_file.name or state.uploaded_image is None:
                    image_bytes = uploaded_file.read()
                    if len(image_bytes) > 0:
                        image = load_image(image_bytes)
                        is_valid, error = validate_image(image)
                        if not is_valid:
                            st.error(f"Invalid image: {error}")
                            st.stop()
                        set_uploaded_image(image, uploaded_file.name, source="upload")
                        state = get_state()  # Refresh state
                        st.success(f"Image loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.stop()

    else:  # AI Generation path
        st.subheader("Generate with AI")

        if not GEMINI_AVAILABLE:
            st.error("The `google-genai` package is not installed.")
            st.stop()

        # Mode selection
        generation_mode = st.radio(
            "Generation mode",
            ["free", "paid"],
            format_func=lambda x: "Use my own API key (Free)" if x == "free"
                                  else "Use credits ($1 for 3 generations)",
            horizontal=True,
        )

        api_key = None
        can_generate = False
        credit_code = None

        if generation_mode == "free":
            api_key = st.text_input("Google AI API Key", type="password",
                                    help="Your key is not stored.")
            st.caption("Get your key at: https://aistudio.google.com/apikey")
            can_generate = bool(api_key)

        else:  # paid
            credit_code = st.text_input("Credit Code", placeholder="e.g., FONT-A7X9K2")

            if credit_code:
                credits = get_credits(credit_code)
                if credits is None:
                    st.error("Invalid code")
                elif credits > 0:
                    st.success(f"Valid! {credits} credit(s) remaining")
                    api_key = st.secrets.get("DEVELOPER_GEMINI_API_KEY")
                    can_generate = True
                else:
                    st.warning("No credits remaining on this code")

            st.divider()
            st.link_button("Buy 3 Credits - $1", st.secrets.get("STRIPE_PAYMENT_LINK", "#"))

        col1, col2 = st.columns(2)

        with col1:
            # Style preset selection
            style_options = ["custom"] + list(STYLE_PRESETS.keys())
            style_preset = st.selectbox(
                "Style Preset",
                options=style_options,
                format_func=lambda x: x.replace("_", " ").title() if x != "custom" else "Custom Style",
            )

        with col2:
            # Character set selection
            char_set_key = st.selectbox(
                "Character Set",
                options=list(CHARACTER_SETS.keys()),
                format_func=lambda x: CHARACTER_SET_LABELS.get(x, x),
            )

        # Custom style input (always shown, but required if preset is "custom")
        custom_style = st.text_input(
            "Custom Style Description",
            placeholder="e.g., elegant cursive with thick downstrokes",
            help="Add your own style description. Will be combined with preset if one is selected.",
        )

        # Show the characters that will be generated
        with st.expander("Characters to generate"):
            st.code(CHARACTER_SETS[char_set_key])

        # Generate button
        if st.button("Generate Glyph Sheet", type="primary", disabled=not can_generate):
            # Build the full style description
            style_desc = get_style_description(style_preset, custom_style)

            # Build the prompt
            prompt = build_prompt(style_desc, char_set_key)

            try:
                with st.spinner("Generating glyph sheet with Imagen 3..."):
                    client = GeminiClient(api_key)
                    image, metadata = client.generate_glyph_sheet(prompt)

                    # Set the image in state (same as upload path)
                    filename = f"ai_generated_{style_preset}_{char_set_key}.png"
                    set_uploaded_image(image, filename, source="ai_generated")
                    state = get_state()  # Refresh state

                    st.success(f"Glyph sheet generated! ({metadata['width']}x{metadata['height']})")

                    # Deduct credit if using paid mode
                    if generation_mode == "paid" and credit_code:
                        use_credit(credit_code)

            except GeminiError as e:
                st.error(f"Generation failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # Show uploaded image and preprocessing options
    if state.uploaded_image is not None:
        image = state.uploaded_image

        info = get_image_info(image)
        col1, col2, col3 = st.columns(3)
        col1.metric("Width", f"{info['width']}px")
        col2.metric("Height", f"{info['height']}px")
        col3.metric("Channels", info['channels'])

        # Display image
        if len(image.shape) == 3:
            display_image = image[:, :, ::-1]
        else:
            display_image = image
        st.image(display_image, caption="Uploaded image", use_column_width=True)

        # Preprocessing
        st.subheader("Preprocessing")
        col1, col2 = st.columns(2)

        with col1:
            auto_invert = auto_detect_invert(image)
            invert = st.checkbox("Invert Image", value=auto_invert)

        with col2:
            denoise = st.checkbox("Remove Noise", value=True)

        # Binarization method
        threshold_method = st.selectbox(
            "Binarization Method",
            ["Automatic (Otsu)", "Adaptive", "Manual"],
            help="How to convert the image to black and white"
        )

        manual_threshold = None
        if threshold_method == "Manual":
            manual_threshold = st.slider("Threshold Value", 0, 255, 128)

        if st.button("Process Image", type="primary"):
            # Determine threshold value based on method
            if threshold_method == "Manual":
                thresh = manual_threshold
            else:
                thresh = None

            processed = preprocess_image(image, threshold=thresh, invert=invert, denoise=denoise)
            set_processed_image(processed)
            st.success("Image processed! Go to the Segment tab.")
            state = get_state()  # Refresh state

        # Show processed image
        if state.processed_image is not None:
            st.image(state.processed_image, caption="Processed (binary)", use_column_width=True)

# ============ TAB 2: SEGMENT ============
with tab2:
    state = get_state()  # Fresh state for this tab
    st.header("Segment Glyphs")

    if state.processed_image is None:
        st.warning("Please upload and process an image first (Upload tab).")
    else:
        # Initialize session state for selections
        if 'selected_glyphs' not in st.session_state:
            st.session_state.selected_glyphs = set()

        st.markdown("Extract individual glyphs using contour detection.")

        col1, col2, col3 = st.columns(3)
        with col1:
            min_area = st.slider("Min Area", 10, 1000, 100)
        with col2:
            min_size = st.slider("Min Size", 5, 100, 10)
        with col3:
            padding = st.slider("Padding", 0, 20, 2)

        merge = st.checkbox("Merge nearby components", value=True)

        if st.button("Extract Glyphs", type="primary"):
            glyphs = segment_glyphs(
                state.processed_image,
                min_area=min_area,
                min_size=min_size,
                padding=padding
            )

            if merge and glyphs:
                glyphs = merge_nearby_contours(glyphs, state.processed_image)

            if glyphs:
                glyph_data_list = []
                for i, (img, bounds) in enumerate(glyphs):
                    glyph_data_list.append(GlyphData(
                        image=img,
                        bounds=(bounds.x, bounds.y, bounds.width, bounds.height),
                        index=i
                    ))
                set_glyphs(glyph_data_list)
                st.session_state.selected_glyphs = set()
                st.success(f"Extracted {len(glyph_data_list)} glyphs!")
            else:
                st.warning("No glyphs found. Adjust parameters.")

        if state.glyphs:
            st.divider()

            # Prepare visualization image
            if len(state.processed_image.shape) == 2:
                viz_image = cv2.cvtColor(state.processed_image, cv2.COLOR_GRAY2BGR)
            else:
                viz_image = state.processed_image.copy()

            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

            # Draw glyph boxes - selected ones get thicker green border
            for glyph in state.glyphs:
                x, y, w, h = glyph.bounds
                is_selected = glyph.index in st.session_state.selected_glyphs
                if is_selected:
                    color = (0, 255, 0)  # Green for selected
                    thickness = 3
                else:
                    color = colors[glyph.index % len(colors)]
                    thickness = 2
                cv2.rectangle(viz_image, (x, y), (x+w, y+h), color, thickness)
                cv2.putText(viz_image, str(glyph.index), (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Show glyph thumbnails with checkboxes for selection
            st.subheader(f"Select Glyphs ({len(state.glyphs)} total)")
            cols_per_row = 10
            for row_start in range(0, len(state.glyphs), cols_per_row):
                row_glyphs = state.glyphs[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for i, glyph in enumerate(row_glyphs):
                    with cols[i]:
                        img = glyph.image
                        h, w = img.shape[:2]
                        scale = min(60/h, 60/w)
                        display = cv2.resize(img, (max(1,int(w*scale)), max(1,int(h*scale))))
                        st.image(display)
                        is_selected = st.checkbox(
                            f"#{glyph.index}",
                            value=glyph.index in st.session_state.selected_glyphs,
                            key=f"sel_{glyph.index}"
                        )
                        if is_selected:
                            st.session_state.selected_glyphs.add(glyph.index)
                        elif glyph.index in st.session_state.selected_glyphs:
                            st.session_state.selected_glyphs.discard(glyph.index)

            # Action buttons
            selected_count = len(st.session_state.selected_glyphs)
            selected_text = f"**Selected: {selected_count}**"
            if selected_count > 0:
                selected_text += f" (#{', #'.join(str(i) for i in sorted(st.session_state.selected_glyphs))})"
            st.markdown(selected_text)

            col1, col2, col3 = st.columns(3)
            with col1:
                merge_clicked = st.button("Merge Selected", disabled=selected_count < 2, type="primary")
            with col2:
                delete_clicked = st.button("Delete Selected", disabled=selected_count < 1, type="secondary")
            with col3:
                clear_clicked = st.button("Clear Selection", disabled=selected_count < 1)

            if merge_clicked:
                merged = merge_selected_glyphs(
                    state.glyphs,
                    list(st.session_state.selected_glyphs),
                    state.processed_image
                )
                set_glyphs(merged)
                st.session_state.selected_glyphs = set()
                st.rerun()

            if delete_clicked:
                remaining = [g for g in state.glyphs if g.index not in st.session_state.selected_glyphs]
                for i, g in enumerate(sorted(remaining, key=lambda x: (x.bounds[1], x.bounds[0]))):
                    g.index = i
                set_glyphs(remaining)
                st.session_state.selected_glyphs = set()
                st.rerun()

            if clear_clicked:
                st.session_state.selected_glyphs = set()
                st.rerun()

            # Glyph locations visualization
            st.divider()
            st.subheader("Glyph Locations")
            st.image(viz_image[:,:,::-1], caption="Detected glyph locations (selected in green)", use_column_width=True)

            # === DRAW CUSTOM BOX SECTION ===
            st.divider()
            st.subheader("Draw Custom Box")
            st.markdown("**Draw a rectangle** on the image below to add a custom glyph region.")

            from streamlit_drawable_canvas import st_canvas

            # Canvas for drawing
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=Image.fromarray(viz_image[:,:,::-1]),
                drawing_mode="rect",
                height=viz_image.shape[0],
                width=viz_image.shape[1],
                key="canvas",
            )

            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                rects = [obj for obj in objects if obj.get("type") == "rect"]
                if rects:
                    st.info(f"{len(rects)} rectangle(s) drawn")
                    if st.button("Add Drawn Boxes as Glyphs", type="primary"):
                        new_glyphs = []
                        for obj in rects:
                            x = int(obj["left"])
                            y = int(obj["top"])
                            w = int(obj["width"] * obj.get("scaleX", 1))
                            h = int(obj["height"] * obj.get("scaleY", 1))
                            # Clamp to image bounds
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, state.processed_image.shape[1] - x)
                            h = min(h, state.processed_image.shape[0] - y)
                            if w > 0 and h > 0:
                                glyph_img = state.processed_image[y:y+h, x:x+w].copy()
                                new_glyphs.append(GlyphData(image=glyph_img, bounds=(x, y, w, h), index=0))

                        if new_glyphs:
                            all_glyphs = state.glyphs + new_glyphs
                            for i, g in enumerate(sorted(all_glyphs, key=lambda x: (x.bounds[1], x.bounds[0]))):
                                g.index = i
                            set_glyphs(all_glyphs)
                            st.rerun()

# ============ TAB 3: MAP ============
with tab3:
    state = get_state()  # Fresh state for this tab
    st.header("Map Characters")

    if not state.glyphs:
        st.warning("Please extract glyphs first (Segment tab).")
    else:
        st.markdown("Assign characters to each glyph.")

        # Auto-assign
        st.subheader("Auto-Assign")
        col1, col2 = st.columns([3, 1])

        charset_options = {
            "Uppercase (A-Z)": UPPERCASE,
            "Lowercase (a-z)": LOWERCASE,
            "Uppercase + Digits": UPPERCASE + DIGITS,
            "All Letters": UPPERCASE + LOWERCASE,
        }

        with col1:
            selected = st.selectbox("Character Set", list(charset_options.keys()))

        with col2:
            if st.button("Auto-Assign"):
                chars = charset_options[selected]
                state.mappings.clear()
                for i, char in enumerate(chars):
                    if i < len(state.glyphs):
                        set_mapping(i, char)
                st.rerun()

        # Manual mapping
        st.subheader("Manual Mapping")
        cols_per_row = 8

        for row_start in range(0, len(state.glyphs), cols_per_row):
            row_glyphs = state.glyphs[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)

            for i, glyph in enumerate(row_glyphs):
                with cols[i]:
                    img = glyph.image
                    h, w = img.shape[:2]
                    scale = min(50/h, 50/w)
                    display = cv2.resize(img, (max(1,int(w*scale)), max(1,int(h*scale))))
                    st.image(display)

                    current = state.mappings.get(glyph.index, "")
                    new_char = st.text_input(
                        f"#{glyph.index}",
                        value=current,
                        max_chars=1,
                        key=f"map_{glyph.index}",
                        label_visibility="collapsed"
                    )

                    if new_char != current:
                        if new_char:
                            set_mapping(glyph.index, new_char)
                        else:
                            remove_mapping(glyph.index)

        # Summary
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Mapped", len(state.mappings))
        col2.metric("Unmapped", len(state.glyphs) - len(state.mappings))

        if state.mappings:
            chars = ''.join(sorted(state.mappings.values()))
            st.code(chars)

# ============ TAB 4: GENERATE ============
with tab4:
    state = get_state()  # Fresh state for this tab
    st.header("Generate Font")

    if not state.glyphs:
        st.warning("Please extract glyphs first (Segment tab).")
    elif not state.mappings:
        st.warning("Please map characters first (Map tab).")
    else:
        # Font mode selection
        font_mode = st.radio(
            "Font Mode",
            options=["monochrome", "color"],
            index=0 if state.font_mode == "monochrome" else 1,
            format_func=lambda x: "Monochrome (TTF/WOFF/WOFF2)" if x == "monochrome" else "Color (OpenType-SVG)",
            help="Monochrome: Standard vector outlines. Color: Embedded color images.",
            horizontal=True,
        )
        state.font_mode = font_mode

        # Check Potrace only for monochrome mode
        potrace_ok = True
        if font_mode == "monochrome":
            potrace_ok, potrace_msg = check_potrace_installed()
            if not potrace_ok:
                st.error(f"Potrace not installed! {potrace_msg}")
                st.markdown("""
                Install Potrace:
                - **macOS:** `brew install potrace`
                - **Ubuntu:** `sudo apt-get install potrace`

                *Or use Color mode which doesn't require Potrace.*
                """)
            else:
                st.success(f"Potrace: {potrace_msg}")
        else:
            st.info("Color mode: Glyphs will be embedded as PNG bitmaps using CBDT/CBLC format (excellent browser support).")

        if potrace_ok or font_mode == "color":
            # Font settings
            col1, col2 = st.columns(2)
            with col1:
                font_family = st.text_input("Font Family Name", value="MyFont")
            with col2:
                style = st.selectbox("Style", ["Regular", "Bold", "Italic"])

            st.metric("Characters to include", len(state.mappings))

            if st.button("Generate Font", type="primary"):
                progress = st.progress(0, "Starting...")

                try:
                    progress.progress(20, "Preparing glyphs...")
                    char_images = {}

                    if font_mode == "color" and state.uploaded_image is not None:
                        # For color mode, re-extract glyphs from original color image
                        # using the bounds from the binary segmentation
                        for glyph in state.glyphs:
                            if glyph.index in state.mappings:
                                char = state.mappings[glyph.index]
                                # Extract from original image using glyph bounds
                                x, y, w, h = glyph.bounds
                                color_glyph = state.uploaded_image[y:y+h, x:x+w].copy()
                                # Normalize background to pure white using the binary mask
                                color_glyph = normalize_background(color_glyph, glyph.image)
                                # Remove background (now guaranteed to be pure white)
                                color_glyph = remove_background(color_glyph)
                                char_images[char] = color_glyph
                    else:
                        # For monochrome mode, use the binary glyph images
                        for glyph in state.glyphs:
                            if glyph.index in state.mappings:
                                char = state.mappings[glyph.index]
                                char_images[char] = glyph.image

                    progress.progress(50, "Building font...")
                    print(f"DEBUG: Building font with mode={font_mode}, {len(char_images)} glyphs")
                    font = build_font(
                        char_images,
                        font_family=font_family,
                        style_name=style,
                        font_mode=font_mode,
                    )
                    print(f"DEBUG: Font built successfully, tables: {list(font.keys())}")

                    progress.progress(80, "Generating files...")
                    st.session_state['generated_font'] = font
                    st.session_state['font_name'] = font_family
                    st.session_state['font_mode'] = font_mode
                    print(f"DEBUG: Stored in session state, font_mode={font_mode}")

                    progress.progress(100, "Done!")
                    st.success("Font generated!")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # Download section
            if 'generated_font' in st.session_state:
                st.divider()
                st.subheader("Download")

                font = st.session_state['generated_font']
                font_name = st.session_state['font_name']
                generated_mode = st.session_state.get('font_mode', 'monochrome')
                print(f"DEBUG: Download section - generated_mode={generated_mode}")

                if generated_mode == "color":
                    print("DEBUG: Entering color download branch")
                    # Color font: CBDT/CBLC format
                    try:
                        cbdt_bytes = get_cbdt_bytes(font)
                        print(f"DEBUG: Got {len(cbdt_bytes)} bytes for color font")
                        st.download_button(
                            "📥 Color TTF (CBDT)",
                            data=cbdt_bytes,
                            file_name=f"{font_name}-color.ttf",
                            mime="font/ttf"
                        )
                        st.caption(f"Size: {len(cbdt_bytes) / 1024:.1f} KB")
                        st.success("CBDT color fonts work in Chrome, Firefox, Edge, and Safari.")
                    except Exception as e:
                        print(f"DEBUG: Error in color download: {e}")
                        import traceback
                        traceback.print_exc()
                        st.error(f"Error preparing download: {e}")
                else:
                    # Monochrome: all formats
                    font_bytes = get_all_formats_bytes(font)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.download_button(
                            "📥 TTF",
                            data=font_bytes['ttf'],
                            file_name=f"{font_name}.ttf",
                            mime="font/ttf"
                        )

                    with col2:
                        st.download_button(
                            "📥 WOFF",
                            data=font_bytes['woff'],
                            file_name=f"{font_name}.woff",
                            mime="font/woff"
                        )

                    with col3:
                        st.download_button(
                            "📥 WOFF2",
                            data=font_bytes['woff2'],
                            file_name=f"{font_name}.woff2",
                            mime="font/woff2"
                        )

                # Preview
                st.subheader("Preview")
                preview_text = st.text_input("Preview text", value="Hello World")

                try:
                    if is_color_font(font):
                        # Use color font preview
                        preview = render_color_font_preview(font, preview_text, font_size=48)
                        st.image(preview)

                        grid = create_color_glyph_grid(font)
                        st.image(grid, caption="Character grid")
                    else:
                        # Use standard preview
                        preview = render_text_preview(font, preview_text, font_size=48)
                        st.image(preview)

                        grid = create_glyph_grid(font)
                        st.image(grid, caption="Character grid")
                except Exception as e:
                    st.warning(f"Preview error: {e}")

# Sidebar status
state = get_state()  # Fresh state for sidebar
st.sidebar.title("Progress")
st.sidebar.checkbox("Image uploaded", value=state.uploaded_image is not None, disabled=True)
st.sidebar.checkbox("Image processed", value=state.processed_image is not None, disabled=True)
st.sidebar.checkbox("Glyphs extracted", value=bool(state.glyphs), disabled=True)
st.sidebar.checkbox("Characters mapped", value=bool(state.mappings), disabled=True)
st.sidebar.checkbox("Font generated", value='generated_font' in st.session_state, disabled=True)
