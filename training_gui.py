import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import random
import minizinc 
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tkinter import simpledialog

def run_training_gui():
    """Function to run the training model GUI."""

    def analyze_data(root):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            messagebox.showinfo("Error", "Please select a dataset.")
            return

        data = pd.read_csv(file_path)
        data['subset_exists'] = data['subset_exists'].astype('category')

        try:
            train_percentage = int(train_percentage_entry.get())
            if not 0 < train_percentage < 100:
                raise ValueError("Percentage must be between 1 and 99.")
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return

        train_data, test_data = train_test_split(data, test_size=(100-train_percentage)/100, random_state=1148, stratify=data['subset_exists'])

        selected_model = dropdown_var1.get()

        if selected_model == "Decision tree model":
            criterion = criterion_var.get()
            max_depth = None if max_depth_entry.get() == '' else int(max_depth_entry.get())
            min_samples_split = int(min_samples_split_entry.get())
            min_samples_leaf = int(min_samples_leaf_entry.get())
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            model_name = "Decision Tree"
        elif selected_model == "Xgboost tree model":
            model = XGBClassifier()
            model_name = "XGBoost"
        elif selected_model == "Random forest model":
            model = RandomForestClassifier(n_estimators=1000, max_features=2)
            model_name = "Random Forest"

        model.fit(train_data.drop('subset_exists', axis=1), train_data['subset_exists'])
        evaluate_model(model, test_data.drop('subset_exists', axis=1), test_data['subset_exists'], model_name)

    def export_model(model, model_name):
        save_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl"), ("Joblib files", "*.joblib")])
        if not save_path:
            messagebox.showerror("Error", "No file selected for saving the model.")
            return

        try:
            if save_path.endswith(".pkl"):
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
            elif save_path.endswith(".joblib"):
                joblib.dump(model, save_path)
            messagebox.showinfo("Success", f"Model saved successfully to {save_path}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def evaluate_model(model, test_features, test_labels, model_name):
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        probabilities = model.predict_proba(test_features)[:, 1]
        fpr, tpr, _ = roc_curve(test_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(test_labels, probabilities)
        average_precision = average_precision_score(test_labels, probabilities)

        plot_window = tk.Toplevel()
        plot_window.title("Analysis Results")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], linestyle='--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {model_name}')
        ax1.legend(loc="lower right")

        ax2.plot(recall, precision, label=f'AP = {average_precision:.2f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {model_name}')
        ax2.legend(loc="lower right")

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        save_button = tk.Button(plot_window, text="Save Plots", command=lambda: save_plots(fig, model_name))
        save_button.pack(side=tk.BOTTOM)

        export_button = tk.Button(plot_window, text="Export Model", command=lambda: export_model(model, model_name))
        export_button.pack(side=tk.RIGHT, padx=10, pady=10)

        messagebox.showinfo("Analysis Complete", f"Analysis displayed in a new window.\nModel Accuracy: {accuracy:.2%}")

    def save_plots(fig, model_name):
        output_dir = 'analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{model_name}_analysis_plots.png'))
        messagebox.showinfo("Save Successful", f"Plots saved in '{output_dir}' folder.")

    def plot_neural_network(model):
        if not model.built:
            model.build(input_shape=(None, model.layers[0].input_shape[-1]))

        layer_names = [layer.name for layer in model.layers]
        layer_shapes = [layer.units if isinstance(layer, Dense) else layer.output_shape[-1] for layer in model.layers]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(layer_names))
        y = np.array(layer_shapes)

        plt.bar(x, y, tick_label=layer_names, color='skyblue')

        for i, val in enumerate(y):
            plt.text(i, val + 1, f'{val} neurons', ha='center', va='bottom')

        plt.title("Neural Network Architecture")
        plt.xlabel("Layers")
        plt.ylabel("Number of Neurons")

        plt.show()

    def run_neural_script():
        root.withdraw()

        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            print("No file selected. Exiting.")
            return

        df = pd.read_csv(file_path)

        encoder = LabelEncoder()
        df['donor_blood_type'] = encoder.fit_transform(df['donor_blood_type'])
        df['recipient_blood_type'] = encoder.transform(df['recipient_blood_type'])

        feature_columns = simpledialog.askstring("Input", "Enter feature columns separated by commas (e.g., 'donor_blood_type, recipient_blood_type')")
        target_column = simpledialog.askstring("Input", "Enter target column (e.g., 'compatibility')")

        if not feature_columns or not target_column:
            print("Feature columns or target column not provided. Exiting.")
            return

        feature_columns = [col.strip() for col in feature_columns.split(',')]

        X = df[feature_columns]
        y = df[target_column]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        epochs = simpledialog.askinteger("Input", "Enter the number of epochs:", initialvalue=50, minvalue=1)
        batch_size = simpledialog.askinteger("Input", "Enter the batch size:", initialvalue=10, minvalue=1)

        if not epochs or not batch_size:
            print("Epochs or batch size not provided. Exiting.")
            return

        model = Sequential([
            Dense(16, input_dim=X_train.shape[1], activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        display_plot = messagebox.askyesno("Input", "Do you want to display the neural network plot?")

        if display_plot:
            plot_neural_network(model)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        loss, accuracy = model.evaluate(X_test, y_test)
        messagebox.showinfo("Model Evaluation", f"Test Accuracy: {accuracy*100:.2f}%")

        save_model = messagebox.askyesno("Input", "Do you want to save the model?")

        if save_model:
            save_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5")])
            if save_path:
                model.save(save_path)
                print(f"Model saved at {save_path}")
            else:
                print("Save operation canceled.")

    root = tk.Tk()
    root.title("Data Analysis Tool")

    frame = tk.Frame(root)
    frame.pack(pady=20, padx=20)

    # Criterion OptionMenu with default value "gini"
    tk.Label(frame, text="Criterion:").grid(row=0, column=0, sticky=tk.W)
    criterion_var = tk.StringVar(value="gini")  # Set default value
    criterion_menu = tk.OptionMenu(frame, criterion_var, "gini", "entropy")  # OptionMenu linked to criterion_var
    criterion_menu.grid(row=0, column=1, sticky=tk.W)

    tk.Label(frame, text="Max Depth:").grid(row=1, column=0, sticky=tk.W)
    max_depth_entry = tk.Entry(frame)
    max_depth_entry.grid(row=1, column=1)

    tk.Label(frame, text="Min Samples Split:").grid(row=2, column=0, sticky=tk.W)
    min_samples_split_entry = tk.Entry(frame)
    min_samples_split_entry.grid(row=2, column=1)

    tk.Label(frame, text="Min Samples Leaf:").grid(row=3, column=0, sticky=tk.W)
    min_samples_leaf_entry = tk.Entry(frame)
    min_samples_leaf_entry.grid(row=3, column=1)

    tk.Label(frame, text="Percentage of instances to train:").grid(row=4, column=0, sticky=tk.W)
    train_percentage_entry = tk.Entry(frame)
    train_percentage_entry.grid(row=4, column=1)

    # Model Type OptionMenu with a default value "Decision tree model"
    tk.Label(frame, text="Select Model:").grid(row=5, column=0, sticky=tk.W)
    dropdown_var1 = tk.StringVar(value="Decision tree model")  # Set default value
    model_menu = tk.OptionMenu(frame, dropdown_var1, "Decision tree model", "Xgboost tree model", "Random forest model")  # OptionMenu linked to dropdown_var1
    model_menu.grid(row=5, column=1, sticky=tk.W)

    analyze_button = tk.Button(root, text="Analyze Data", command=lambda: analyze_data(root))
    analyze_button.pack(padx=10, pady=20)

    # Add the Run Neural Script button
    neural_button = tk.Button(root, text="Run Neural Script", command=run_neural_script)
    neural_button.pack(padx=10, pady=5)

    root.mainloop()
