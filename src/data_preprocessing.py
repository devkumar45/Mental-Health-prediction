import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_data(path="data/mental_health_data.csv"):
  
    # Check if the dataset exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found at path: {path}")

    print(f"📂 Loading dataset from: {path}")
    df = pd.read_csv(path)
    
    # Show basic info
    print(f"✅ Dataset loaded successfully! Total Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Drop missing values for cleaner training
    before = df.shape[0]
    df = df.dropna()
    after = df.shape[0]

    print(f"🧹 Dropped {before - after} missing rows. Cleaned Data Rows: {after}")
    return df


def preprocess_and_save(df, scaler_path="models/scaler.pkl", le_path="models/label_encoder.pkl"):
    """
    Encodes labels, scales numerical features, and saves preprocessors.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        scaler_path (str): Path to save the StandardScaler model.
        le_path (str): Path to save the LabelEncoder model.
    
    Returns:
        tuple: Scaled features, encoded labels, label encoder, and scaler.
    """

    print("\n🔄 Starting preprocessing...")

    # Encode target label (Low, Medium, High → 0, 1, 2)
    if 'Stress_Level' not in df.columns:
        raise KeyError("❌ 'Stress_Level' column not found in dataset!")

    le = LabelEncoder()
    df['Stress_Label'] = le.fit_transform(df['Stress_Level'])

    # Select features and target
    feature_columns = ['Sleep_Hours', 'Study_Hours', 'Screen_Time', 
                       'Social_Activity', 'Diet_Quality', 'Exercise']
    
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"❌ Missing feature columns in dataset: {missing_features}")

    X = df[feature_columns]
    y = df['Stress_Label']

    print("🧾 Selected Features:", feature_columns)
    print("🎯 Target Variable: Stress_Label")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create model directory if not exists
    os.makedirs("models", exist_ok=True)

    # Save the scaler and label encoder
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)

    print(f"💾 Scaler saved to: {scaler_path}")
    print(f"💾 Label Encoder saved to: {le_path}")

    print("✅ Preprocessing completed successfully!")
    print(f"📊 Scaled Data Shape: {X_scaled.shape}")

    return X_scaled, y, le, scaler


# =====================================================
# 🚀 Step 3: Execute when run directly
# =====================================================
if __name__ == "__main__":
    try:
        print("\n🧠 Running Mental Health Data Preprocessing...\n")
        df = load_data()  # Load the dataset
        X_scaled, y, le, scaler = preprocess_and_save(df)
        print("\n🎉 All preprocessing steps completed successfully!")
        print(f"📈 Total Samples Processed: {X_scaled.shape[0]}")
        print("------------------------------------------------")
    except Exception as e:
        print(f"\n❌ Error Occurred: {e}")
        print("⚠️ Please check your file paths or data format again.")
