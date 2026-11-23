# ============================================================
# CNN MINOR PROJECT: PART A + PART B
# Part A: MNIST CNN Image Classification
# Part B: Student Marks Calculator (Streamlit)
# ============================================================

import streamlit as st

st.set_page_config(page_title="CNN Minor Project", page_icon="📚", layout="centered")

st.title("CNN Image Classification and Streamlit Application")
st.write("Part A: MNIST CNN Image Classification | Part B: Student Marks Calculator")

# ============================================================
# PART A - CNN IMAGE CLASSIFICATION
# ============================================================

def run_part_a():
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from contextlib import redirect_stdout
    from io import StringIO

    st.header("Part A - CNN Image Classification")
    st.write("This section trains a CNN model using the MNIST handwritten digit dataset.")

    with st.spinner("Loading MNIST dataset and training the CNN model..."):
        # Load MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        st.subheader("1. Dataset Information")
        st.write("Training images:", X_train.shape)
        st.write("Training labels:", y_train.shape)
        st.write("Testing images:", X_test.shape)
        st.write("Testing labels:", y_test.shape)
        st.write("Number of classes:", len(np.unique(y_train)))
        st.write("Class labels:", np.unique(y_train))

        # Display sample MNIST images
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes.flat):
            ax.imshow(X_train[i], cmap="gray")
            ax.set_title("Label: " + str(y_train[i]))
            ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Preprocess dataset
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        st.subheader("2. Preprocessing")
        st.write("Training data shape:", X_train.shape)
        st.write("Testing data shape:", X_test.shape)
        st.write("Minimum pixel value:", float(X_train.min()))
        st.write("Maximum pixel value:", float(X_train.max()))

        # Build CNN model
        model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax")
        ])

        st.subheader("3. CNN Model Summary")
        summary_buffer = StringIO()
        with redirect_stdout(summary_buffer):
            model.summary()
        st.code(summary_buffer.getvalue())

        # Compile and train
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        st.subheader("4. Model Training")
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=64,
            validation_split=0.2,
            verbose=0
        )
        st.success("CNN model training completed.")

        # Evaluate model
        st.subheader("5. Model Evaluation")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")
        st.metric("Test Loss", f"{test_loss:.4f}")

        # Accuracy plot
        st.subheader("6. Training and Validation Accuracy")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["accuracy"], label="Training Accuracy")
        ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax.set_title("CNN Model Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        plt.close(fig)

        # Loss plot
        st.subheader("7. Training and Validation Loss")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["loss"], label="Training Loss")
        ax.plot(history.history["val_loss"], label="Validation Loss")
        ax.set_title("CNN Model Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        plt.close(fig)

        # Predictions
        st.subheader("8. Actual and Predicted Labels")
        predictions = model.predict(X_test[:10], verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
            ax.set_title(
                f"Actual: {y_test[i]}\\nPredicted: {predicted_labels[i]}"
            )
            ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Correct and incorrect predictions
        all_predictions = model.predict(X_test, verbose=0)
        predicted_all = np.argmax(all_predictions, axis=1)
        correct = int(np.sum(predicted_all == y_test))
        incorrect = int(np.sum(predicted_all != y_test))

        st.subheader("9. Prediction Results")
        st.write("Correct Predictions:", correct)
        st.write("Incorrect Predictions:", incorrect)
        st.write("Total Predictions:", len(y_test))

    st.success("Part A CNN workflow completed successfully.")


with st.expander("Part A - Run CNN Image Classification", expanded=False):
    st.write("Click the button below to load MNIST, train the CNN, evaluate it, and display predictions.")
    if st.button("Run Part A - Train and Evaluate CNN"):
        run_part_a()

st.divider()

# ============================================================
# PART B - STREAMLIT APPLICATION
# ============================================================

st.header("Part B - Student Marks Calculator")
st.write("Enter student details and marks to calculate total marks, percentage, grade, and result.")

name = st.text_input("Enter Student Name", key="student_name")

subjects = [
    "Python",
    "Mathematics",
    "Computer Science",
    "English",
    "Artificial Intelligence"
]

marks = []
for subject in subjects:
    mark = st.number_input(
        subject,
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        key=f"mark_{subject}"
    )
    marks.append(mark)

if st.button("Calculate Results", type="primary"):
    if not name.strip():
        st.error("Please enter your name.")
    else:
        total = sum(marks)
        percentage = total / len(subjects)

        if percentage >= 90:
            grade = "A+"
        elif percentage >= 80:
            grade = "A"
        elif percentage >= 70:
            grade = "B"
        elif percentage >= 60:
            grade = "C"
        elif percentage >= 50:
            grade = "D"
        else:
            grade = "F"

        st.success("Calculation Completed!")
        st.write("Student Name:", name.strip())
        col1, col2 = st.columns(2)
        col1.metric("Total Marks", f"{total}/500")
        col2.metric("Percentage", f"{percentage:.2f}%")
        st.metric("Grade", grade)

        if all(mark >= 35 for mark in marks):
            st.success("Result: PASS")
        else:
            st.error("Result: FAIL")

st.caption("CNN Minor Project - Part A and Part B")
