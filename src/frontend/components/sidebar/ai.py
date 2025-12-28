import streamlit as st
import os
import concurrent.futures
from src.backend.ai.data_manager import get_dataset_stats
from src.backend.ai.trainer import train_model, MODELS_DIR
from src.backend.ai.predictor import predict_settings
from src.backend.ai.features import extract_features
from src.frontend.state import save_settings

def apply_prediction(current_file_name: str, model_name: str):
    """
    Callback to apply AI predictions to the current session state.
    """
    if 'preview_raw' in st.session_state:
        try:
            # Extract features
            feats = extract_features(st.session_state.preview_raw)
            # Predict
            preds = predict_settings(feats, model_name)
            # Apply to session state
            for k, v in preds.items():
                st.session_state[k] = float(v)
                # Handle Black/White points special case
                if k == 'black_point':
                    # Need to check if bw_points tuple exists
                    curr_w = st.session_state.get('bw_points', (0.0, 1.0))[1]
                    st.session_state.bw_points = (float(v), curr_w)
                elif k == 'white_point':
                    curr_b = st.session_state.get('bw_points', (0.0, 1.0))[0]
                    st.session_state.bw_points = (curr_b, float(v))
            
            save_settings(current_file_name)
            st.toast("AI settings applied!")
        except Exception as e:
            st.toast(f"Prediction failed: {e}")
    else:
        st.toast("No image loaded.")

def render_ai_tab(current_file_name: str):
    """
    Renders the AI Assistant tab in the sidebar.
    """
    st.header("🤖 AI Assistant")
    
    # 1. Data Collection Control
    st.subheader("1. Learn")
    st.caption("Collect data from your edits to train the model.")
    
    st.checkbox("Collect Training Data on Save", key="collect_training_data", value=True,
                help="When enabled, every adjustment you make (and save) contributes to the training dataset.")
    
    stats = get_dataset_stats()
    st.text(f"Samples Collected: {stats['count']}")
    
    # 2. Training
    st.subheader("2. Train")
    model_name_input = st.text_input("Model Name", value="my_style")
    
    if st.button("Train Model", width="stretch"):
        with st.spinner("Training model... this may take a moment."):
            # Run in thread/process to avoid blocking UI too hard (though streamlit blocks anyway)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(train_model, model_name_input)
                result = future.result()
            st.success(result)
            st.rerun()

    # 3. Inference
    st.subheader("3. Apply")
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    
    if not models:
        st.info("No trained models found. Train one first!")
    else:
        selected_model = st.selectbox("Select Model", models)
        
        c1, c2 = st.columns(2)
        c1.button("Predict & Apply", type="primary", width="stretch",
                  on_click=apply_prediction, args=(current_file_name, selected_model))
