
import sys
import time
import base64
import hashlib
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
assert (PROJECT_ROOT / "modelExperiments" / "DDR").exists(), (
    "Run streamlit_app.py from the ai/ folder (modelExperiments/DDR not found)."
)
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import streamlit_utils as U
import history_db as db
import xai_viz as V
import llm_interpret as LLM

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False


st.set_page_config(page_title="RetinaXAI Demo", layout="wide")
U.ensure_output_dirs()
db.init_db(U.DB_PATH)


def _to_png_bytes(image) -> bytes:
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        ok, buf = cv2.imencode(".png", arr)
    else:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def show_image(container, image, caption=None):
    try:
        b64 = base64.b64encode(_to_png_bytes(image)).decode("ascii")
        html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;height:auto;border-radius:4px;" />'
        )
        if caption:
            html += (
                f'<div style="text-align:center;color:#888;font-size:0.85em;'
                f'margin-top:2px;">{caption}</div>'
            )
        container.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        container.warning(f"Could not display image: {e}")


def render_txt_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return

    try:
        metrics = U.parse_txt_metrics(text)
    except Exception:
        metrics = []

    skip_metric_tiles = path.name in (
        "fidelity_stability_comparison_DDR_test.txt",
        "slic_insertion_deletion_comparison_DDR_test.txt",
    )

    if metrics and not skip_metric_tiles:
        recognized = any(
            U.metric_family(label) in ("Fidelity", "Stability",
                                       "Deletion AUC", "Insertion AUC")
            for label, _ in metrics
        )
        if recognized:
            real = [(l, v) for l, v in metrics if U.metric_family(l) != "Other"]
            for fam, items in U.group_metrics_by_family(real).items():
                st.markdown(f"**{fam}**")
                for start in range(0, len(items), 4):
                    chunk = items[start:start + 4]
                    cols = st.columns(len(chunk))
                    for col, (label, value) in zip(cols, chunk):
                        col.metric(label, U.fmt3(value))
        else:
            shown = metrics[:12]
            cols = st.columns(min(4, len(shown)))
            for i, (label, value) in enumerate(shown):
                cols[i % len(cols)].metric(label, U.fmt3(value))

    with st.expander(f"Raw text - {path.name}"):
        st.code(text)


CHART_METHOD_NAMES = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++",
                      "xgradcam": "X-Grad-CAM"}
CHART_COLORS = ["#4C78A8", "#F58518", "#54A24B"]

_FIDELITY_GROUPS = {
    "Top 5%": ("fidelity_top5_mean", "fidelity_top5_std"),
    "Top 10%": ("fidelity_top10_mean", "fidelity_top10_std"),
    "Top 15%": ("fidelity_top15_mean", "fidelity_top15_std"),
    "Top 20%": ("fidelity_top20_mean", "fidelity_top20_std"),
}
_STABILITY_GROUPS = {
    "Correlation": ("stab_corr_mean", "stab_corr_std"),
    "SSIM": ("stab_ssim_mean", "stab_ssim_std"),
}
_SLIC_GROUPS = {
    "Deletion AUC": ("deletion_mean", "deletion_std"),
    "Insertion AUC": ("insertion_mean", "insertion_std"),
}

CHART_REPORTS = {
    "fidelity_stability_comparison_DDR_test.csv": [
        ("Fidelity - average",
         "Shows how well the highlighted region explains the prediction. "
         "Higher values indicate better fidelity. Results are averaged across all test images at "
         "5%, 10%, 15%, and 20% of the most important pixels.",
         _FIDELITY_GROUPS, 0),
        ("Fidelity - variability",
         "Shows how much the fidelity scores vary across test images. "
         "Lower values indicate more consistent results.",
         _FIDELITY_GROUPS, 1),
        ("Stability - average",
         "Shows how consistent the explanation remains after small input changes. "
         "Higher values indicate better stability. Results are averaged across all test images.",
         _STABILITY_GROUPS, 0),
        ("Stability - variability",
         "Shows how much the stability scores vary across test images. "
         "Lower values indicate more consistent results.",
         _STABILITY_GROUPS, 1),
    ],
    "slic_insertion_deletion_comparison_DDR_test.csv": [
        ("Insertion / Deletion AUC - average",
         "Deletion AUC measures how quickly the prediction decreases when important pixels are removed. "
         "Lower values are better. Insertion AUC measures how quickly the prediction recovers when important pixels are added back. "
         "Higher values are better. Results are averaged across all test images.",
         _SLIC_GROUPS, 0),
        ("Insertion / Deletion AUC - variability",
         "Shows how much the AUC scores vary across test images. "
         "Lower values indicate more consistent results.",
         _SLIC_GROUPS, 1),
    ],
}


def _grouped_bar(df, methods, title, help_text, groups, stat):
    import numpy as np
    import matplotlib.pyplot as plt
    labels = list(groups)
    x = np.arange(len(labels))
    n = len(methods)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 3.6))
    for i, method in enumerate(methods):
        values = [df.iloc[i][groups[g][stat]] for g in labels]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width,
                      label=method, color=CHART_COLORS[i % len(CHART_COLORS)])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=n, frameon=False, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(help_text)
    plt.close(fig)


def render_comparison_chart(df: pd.DataFrame, charts) -> bool:
    needed = ["method"] + [c for _, _, groups, _ in charts
                           for pair in groups.values() for c in pair]
    if not all(c in df.columns for c in needed):
        return False

    methods = list(df["method"].map(lambda m: CHART_METHOD_NAMES.get(m, m)))
    for title, help_text, groups, stat in charts:
        _grouped_bar(df, methods, title, help_text, groups, stat)

    with st.expander("Show raw numbers"):
        st.dataframe(df.round(3), use_container_width=True)
    return True


def render_significance_chart(df: pd.DataFrame) -> bool:
    needed = ["metric", "method_a", "method_b", "winner", "significant", "t_pvalue"]
    if not all(c in df.columns for c in needed):
        return False

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    def _is_sig(v):
        return str(v).strip().lower() in ("true", "1", "yes", "da")

    def _fmt_p(p):
        try:
            p = float(p)
        except (TypeError, ValueError):
            return str(p)
        return "p<0.001" if p < 0.001 else f"p={p:.3f}"

    nm = CHART_METHOD_NAMES
    winner_color = {"gradcam": CHART_COLORS[0], "gradcampp": CHART_COLORS[1],
                    "xgradcam": CHART_COLORS[2]}

    metrics = list(dict.fromkeys(df["metric"]))
    pairs = list(dict.fromkeys(zip(df["method_a"], df["method_b"])))
    nrows, ncols = len(metrics), len(pairs)

    fig, ax = plt.subplots(figsize=(2.2 * ncols + 2, 0.72 * nrows + 2))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()

    for r, metric in enumerate(metrics):
        for c, (a, b) in enumerate(pairs):
            sub = df[(df["metric"] == metric) & (df["method_a"] == a) &
                     (df["method_b"] == b)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            sig = _is_sig(row["significant"])
            win = str(row["winner"])
            face = winner_color.get(win, "#888888") if sig else "#E6E6E6"
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face,
                                   edgecolor="white", linewidth=2))
            if sig:
                label = f"{nm.get(win, win)}\n{_fmt_p(row['t_pvalue'])}"
                txt_color = "white"
            else:
                label = "no significant\ndifference"
                txt_color = "#555555"
            ax.text(c + 0.5, r + 0.5, label, ha="center", va="center",
                    fontsize=8, color=txt_color)

    ax.set_xticks([c + 0.5 for c in range(ncols)])
    ax.set_xticklabels([f"{nm.get(a, a)}\nvs\n{nm.get(b, b)}" for a, b in pairs],
                       fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks([r + 0.5 for r in range(nrows)])
    ax.set_yticklabels(metrics, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [Patch(facecolor=winner_color[k], label=nm[k])
               for k in ("gradcam", "gradcampp", "xgradcam") if k in winner_color]
    handles.append(Patch(facecolor="#E6E6E6", label="no significant difference"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04),
              ncol=2, frameon=False, fontsize=8, title="Winning method")

    st.markdown("**Which method is significantly better per metric?**")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Each cell compares two methods on one metric. A coloured cell means the "
        "difference is statistically significant (paired test, Bonferroni-"
        "corrected) and is tinted by the winning method; grey means no significant "
        "difference."
    )
    plt.close(fig)

    with st.expander("Show raw numbers"):
        st.dataframe(df.round(4), use_container_width=True)
    return True


def render_csv_file(path: Path):
    try:
        df = pd.read_csv(path)
        df = U.drop_sam_columns(df)
        st.caption(path.name)
        charts = CHART_REPORTS.get(path.name)
        if charts and render_comparison_chart(df, charts):
            return
        if path.name == "significance_test_DDR_test.csv" and render_significance_chart(df):
            return
        st.dataframe(df.round(3), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not parse {path.name} as a table ({e}); showing raw text.")
        try:
            with st.expander(f"Raw text - {path.name}"):
                st.code(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass


def render_png_file(path: Path):
    show_image(st, path, caption=path.name)


def render_category_metrics(metric_values: dict, container=None):
    target = container or st
    by_cat = V.present_metrics_by_category(metric_values)
    if not by_cat:
        target.info("No categorised metrics available for this item.")
        return
    for cat, rows in by_cat.items():
        target.markdown(f"**{cat}**")
        target.caption(V.CATEGORY_LEGENDS[cat])
        fig = V.category_bar_figure(rows)
        target.pyplot(fig)
        import matplotlib.pyplot as _plt
        _plt.close(fig)

        target.write("")
        target.write("")


def llm_interpretation_block(result: dict, *, row_id=None, cache_key=None):
    st.subheader("LLM interpretation (MedGemma, local)")
    st.caption(
        "Uses a local medical LLM to explain the predicted stage and XAI metrics. "
        "The image is not analyzed. It is not a diagnosis."
    )

    key = cache_key or f"llm_{id(result)}"
    existing = result.get("llm_interpretation") or st.session_state.get(key)

    with st.expander("Show exactly what is sent to the model"):
        st.code(LLM.build_prompt_payload(result))

    if st.button("Interpret with LLM", key=f"btn_{key}"):
        with st.spinner("Loading MedGemma and generating (slow on CPU)…"):
            llm, err = LLM.load_llm()
            if err:
                st.error(err)
            else:
                try:
                    text = LLM.interpret(llm, result)
                    st.session_state[key] = text
                    existing = text
                    if row_id is not None:
                        db.update_llm_interpretation(U.DB_PATH, row_id, text)
                        st.success(f"Interpretation saved to history (id={row_id}).")
                except Exception as e:
                    st.error(f"LLM interpretation failed: {e}")

    if existing:
        st.markdown(existing)


def dashboard_section():
    st.header("Results dashboard - dataset-level (whole test set)")

    files = U.scan_report_files()
    if not files:
        st.info("No report files found under modelExperiments/DDR/outputs* or reports*.")
        return

    def _dashboard_legend(section_title: str):
        t = section_title.lower()
        if "fidelity" in t or "stability" in t:
            return V.CATEGORY_LEGENDS["Stability"] + "  " + V.CATEGORY_LEGENDS["Fidelity"]
        if "insertion" in t or "deletion" in t:
            return V.CATEGORY_LEGENDS["Fidelity"]
        if "localization" in t or "dice" in t or "iou" in t:
            return V.CATEGORY_LEGENDS["Localisation"]
        return None

    buckets = U.categorize_files(files)
    for title, bucket_files in buckets.items():
        if not bucket_files:
            continue
        st.subheader(title)
        legend = _dashboard_legend(title)
        if legend:
            st.caption(legend)
        for f in bucket_files:
            suffix = f.suffix.lower()
            if suffix == ".txt":
                render_txt_file(f)
            elif suffix == ".csv":
                render_csv_file(f)
            elif suffix == ".png":
                render_png_file(f)
        st.write("")
        st.write("")        


def _file_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()[:12]


def run_prediction(predictor, image_bytes, method):
    input_tensor, original_image = predictor.preprocess_image(image_bytes)
    idx, conf, prob_map = U.predict_class(predictor, input_tensor)

    cam_obj = U.get_cam_object(predictor, method)
    cam, _, _ = cam_obj.generate(
        input_tensor=input_tensor,
        class_idx=idx,
        image_size=U.IMAGE_SIZE,
    )

    mask = U.retina_mask(original_image)
    cam_vis = U.mask_cam_to_retina(cam, mask)

    heatmap = U.make_heatmap(cam_vis)
    overlay_png = U.overlay_bytes(predictor, original_image, cam_vis)

    metrics = U.compute_image_xai_metrics(
        predictor, cam_obj, original_image, cam, idx, conf,
    )

    return {
        "original_image": original_image,
        "heatmap": heatmap,
        "overlay_png": overlay_png,
        "cam": cam,
        "predicted_index": idx,
        "predicted_class": predictor.class_names[idx],
        "confidence": conf,
        "probabilities": prob_map,
        "method": method,
        "fidelity_top5": metrics.get("fidelity_top5", 0.0),
        "fidelity_top10": metrics.get("fidelity_top10", 0.0),
        "fidelity_top15": metrics.get("fidelity_top15", 0.0),
        "fidelity_top20": metrics.get("fidelity_top20", 0.0),
        "stability": metrics["stability"],
        "stability_corr": metrics["stability_corr"],
        "deletion_auc": metrics["deletion_auc"],
        "insertion_auc": metrics["insertion_auc"],
    }


def live_section():
    st.header("Live prediction - single uploaded image")

    predictor = U.load_predictor()

    uploaded = st.file_uploader("Upload a retina image", type=["png", "jpg", "jpeg"])
    method = st.radio("Explainability method",
                      ["Grad-CAM", "Grad-CAM++", "X-Grad-CAM"], horizontal=True)

    if uploaded is None:
        st.info("Upload a retina image to run a prediction.")
        return

    image_bytes = uploaded.getvalue()
    fhash = _file_hash(image_bytes)
    result_key = f"result_{fhash}_{method}"

    time_key = f"predtime_{fhash}_{method}"

    if st.button("Run prediction"):
        with st.spinner("Running model + Grad-CAM + XAI metrics (AUC is slow)…"):
            _t0 = time.perf_counter()
            st.session_state[result_key] = run_prediction(predictor, image_bytes, method)
            st.session_state[time_key] = time.perf_counter() - _t0
            st.session_state.pop(f"saved_{fhash}_{method}", None)

    result = st.session_state.get(result_key)
    if result is None:
        st.info("Press **Run prediction** to analyse the uploaded image.")
        return

    st.subheader("Metrics for THIS image")
    elapsed = st.session_state.get(time_key)
    if elapsed is not None:
        st.caption(f"Prediction took {elapsed:.2f} s")
    c1, c2 = st.columns(2)
    c1.metric("Predicted class", result["predicted_class"])
    c2.metric("Confidence", f"{result['confidence']:.3f}")

    st.bar_chart(pd.Series(result["probabilities"], name="probability"))

    col_a, col_b, col_c = st.columns(3)
    show_image(col_a, (result["original_image"] * 255).astype(np.uint8),
               caption="Original (512x512)")
    show_image(col_b, result["heatmap"], caption=f"{method} heatmap")
    show_image(col_c, result["overlay_png"], caption="Overlay")

    st.subheader("Explanation quality (this image), by category")
    st.caption(
        "Each bar shows the value on its theoretical scale with the 'better' "
        "direction marked - no good/medium/bad thresholds, since the XAI "
        "literature does not define absolute ones for these metrics. "
        "Fidelity uses the same 5/10/15/20% definition as the dashboard, so each "
        "value here is one sample of that distribution (not clipped, so it may be "
        "slightly negative)."
    )
    render_category_metrics(result)

    annotation_block(predictor, uploaded.name, image_bytes, fhash, result, method)


def annotation_block(predictor, filename, image_bytes, fhash, result, method):
    st.subheader("Doctor annotation (optional)")
    is_doctor = st.checkbox("Are you a doctor and do you want to annotate the image?")

    keep_top_pct = 10.0
    notes = ""

    doctor_mask = None
    dice = iou = None

    if is_doctor:
        keep_top_pct = st.slider("CAM keep-top % (for Dice/IoU binarization)",
                                 min_value=5.0, max_value=25.0, value=10.0, step=1.0)
        notes = st.text_area("Doctor observations (optional)", value="")

        if not CANVAS_AVAILABLE:
            st.error(
                "streamlit-drawable-canvas is not installed. Install it "
                "(see requirements_streamlit.txt) to draw annotations."
            )
        else:
            st.caption("Draw the lesion region. Canvas is locked to 512x512.")
            bg = (result["original_image"] * 255).astype(np.uint8)
            from PIL import Image
            canvas = st_canvas(
                fill_color="rgba(255, 255, 0, 0.4)",
                stroke_width=18,
                stroke_color="rgba(255, 255, 0, 1.0)",
                background_image=Image.fromarray(bg),
                update_streamlit=True,
                height=U.IMAGE_SIZE,
                width=U.IMAGE_SIZE,
                drawing_mode="freedraw",
                key=f"canvas_{fhash}",
            )

            warm_key = f"canvas_warm_{fhash}"
            if not st.session_state.get(warm_key):
                st.session_state[warm_key] = True
                st.rerun()

            if canvas.image_data is not None:
                alpha = canvas.image_data[:, :, 3]
                mask = (alpha > 0).astype(np.uint8)
                if mask.shape != (U.IMAGE_SIZE, U.IMAGE_SIZE):
                    mask = cv2.resize(mask, (U.IMAGE_SIZE, U.IMAGE_SIZE),
                                      interpolation=cv2.INTER_NEAREST)
                doctor_mask = mask

                if doctor_mask.sum() == 0:
                    st.info("Doctor mask is empty - Dice/IoU will be skipped.")
                else:
                    cam_bin = U.cam_to_binary(result["cam"], keep_top_percent=keep_top_pct)
                    dice, iou = U.dice_iou(cam_bin, doctor_mask)
                    m1, m2 = st.columns(2)
                    m1.metric("Dice (this image)", f"{dice:.3f}")
                    m2.metric("IoU (this image)", f"{iou:.3f}")

    result_for_llm = dict(result)
    result_for_llm["dice"] = dice
    result_for_llm["iou"] = iou
    result_for_llm["doctor_notes"] = notes.strip() or None

    llm_key = f"llm_{fhash}_{method}"
    llm_interpretation_block(result_for_llm, cache_key=llm_key)

    saved_key = f"saved_{fhash}_{method}"
    if st.button("Save to history"):
        if st.session_state.get(saved_key):
            st.warning("This prediction was already saved to history.")
        else:
            llm_text = st.session_state.get(llm_key)
            row_id = save_to_history(
                filename, result, method, keep_top_pct,
                dice, iou, doctor_mask, notes, llm_text,
            )
            st.session_state[saved_key] = True
            st.success(f"Saved to history (id={row_id}).")


def save_to_history(filename, result, method, keep_top_pct,
                    dice, iou, doctor_mask, notes, llm_text=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(filename).stem
    base = f"{ts}_{stem}"

    img_path = U.UPLOADS_DIR / f"{base}.png"
    bgr = cv2.cvtColor((result["original_image"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_path), bgr)

    overlay_path = U.OVERLAYS_DIR / f"{base}_overlay.png"
    overlay_path.write_bytes(result["overlay_png"])

    mask_rel = None
    if doctor_mask is not None and doctor_mask.sum() > 0:
        mask_path = U.MASKS_DIR / f"{base}_mask.png"
        cv2.imwrite(str(mask_path), (doctor_mask * 255).astype(np.uint8))
        mask_rel = U.rel_to_outputs(mask_path)

    record = {
        "timestamp": ts,
        "image_filename": filename,
        "predicted_class": result["predicted_class"],
        "confidence": round(float(result["confidence"]), 4),
        "xai_method": method,
        "cam_keep_top_pct": float(keep_top_pct),
        "dice": round(float(dice), 4) if dice is not None else None,
        "iou": round(float(iou), 4) if iou is not None else None,
        "fidelity_top5": result.get("fidelity_top5"),
        "fidelity_top10": result.get("fidelity_top10"),
        "fidelity_top15": result.get("fidelity_top15"),
        "fidelity_top20": result.get("fidelity_top20"),
        "stability": result.get("stability"),
        "stability_corr": result.get("stability_corr"),
        "deletion_auc": result.get("deletion_auc"),
        "insertion_auc": result.get("insertion_auc"),
        "image_path": U.rel_to_outputs(img_path),
        "overlay_path": U.rel_to_outputs(overlay_path),
        "doctor_mask_path": mask_rel,
        "doctor_notes": notes.strip() or None,
        "llm_interpretation": llm_text or None,
    }
    return db.insert_prediction(U.DB_PATH, record)


def history_section():
    st.header("Prediction history")
    rows = db.fetch_all(U.DB_PATH)
    if not rows:
        st.info("No predictions saved yet. Use 'Save to history' on the Live tab.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    ids = [r["id"] for r in rows]
    selected = st.selectbox("View a saved prediction", ids,
                            format_func=lambda i: f"id {i}")
    record = db.fetch_one(U.DB_PATH, selected)
    if not record:
        return

    st.subheader(f"Prediction #{record['id']}")

    h1, h2, h3 = st.columns(3)
    h1.metric("Predicted class", record["predicted_class"])
    h2.metric("Confidence", f"{record['confidence']:.3f}")
    h3.metric("XAI method", record["xai_method"])

    st.markdown("#### Explanation quality, by category")
    render_category_metrics(record)

    if record["dice"] is None:
        st.caption("No doctor annotation was saved for this prediction.")

    img_cols = st.columns(3)
    _show_saved_image(img_cols[0], record.get("image_path"), "Original")
    _show_saved_image(img_cols[1], record.get("overlay_path"), "Overlay")
    _show_saved_image(img_cols[2], record.get("doctor_mask_path"), "Doctor mask")

    if record.get("doctor_notes"):
        st.text_area("Doctor observations", value=record["doctor_notes"], disabled=True)

    llm_interpretation_block(
        record, row_id=record["id"],
        cache_key=f"llm_hist_{record['id']}",
    )


def _show_saved_image(col, rel_path, caption):
    if not rel_path:
        return
    path = U.abs_from_outputs(rel_path)
    if path.exists():
        show_image(col, path, caption=caption)
    else:
        col.warning(f"Missing file: {rel_path}")


def main():
    st.sidebar.title("RetinaXAI")
    page = st.sidebar.radio(
        "Section",
        ["Results dashboard", "Live prediction", "Prediction history"],
    )

    if page == "Results dashboard":
        dashboard_section()
    elif page == "Live prediction":
        live_section()
    else:
        history_section()


if __name__ == "__main__":
    main()