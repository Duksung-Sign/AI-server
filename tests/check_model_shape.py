import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model("sign_language_lstm_model_none.h5",compile=False)

# 요약 출력
model.summary()
