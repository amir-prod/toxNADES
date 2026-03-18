import os
import glob
import pickle
import pandas as pd
from sklearn.metrics import make_scorer
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import shap

warnings.filterwarnings('ignore')


class ModelHandler:
    def __init__(self, folder_dir: str, classification: bool = True):
        self.folder_dir = folder_dir
        self.model_name = folder_dir.split('/')[-1]
        self.model = None
        self.classification = classification



    


    def load_model(self):
        # Find all .pkl files in the folder
        pkl_files = glob.glob(os.path.join(self.folder_dir, '*.pkl'))
        
        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No .pkl file found in {self.folder_dir}")
        elif len(pkl_files) > 1:
            raise ValueError(f"Multiple .pkl files found in {self.folder_dir}: {pkl_files}")
        
        # Open the single .pkl file
        with open(pkl_files[0], 'rb') as f:
            self.model = pickle.load(f)


    def load_data(self):
        self.descriptor_list = self.model.feature_names_in_.tolist()
        self.X_train = pd.read_csv(os.path.join(self.folder_dir, 'X_train.csv'))[self.descriptor_list]
        self.y_train = pd.read_csv(os.path.join(self.folder_dir, 'y_train.csv'))
        self.X_test = pd.read_csv(os.path.join(self.folder_dir, 'X_test.csv'))[self.descriptor_list]
        self.y_test = pd.read_csv(os.path.join(self.folder_dir, 'y_test.csv'))
        print(f"model descriptor list: {self.descriptor_list}")
        print(f"training set size: {len(self.X_train)}")
        print(f"test set size: {len(self.X_test)}")
        

    def get_cv_score_clf(self, model: object, scorer: str):
        cv_scores_list = []
        if scorer == 'matthews_corrcoef':
            scorer = make_scorer(metrics.matthews_corrcoef)
        elif scorer == 'accuracy':
            scorer = make_scorer(metrics.accuracy_score)
        elif scorer == 'precision':
            scorer = make_scorer(metrics.precision_score)
        elif scorer == 'recall':
            scorer = make_scorer(metrics.recall_score)
        elif scorer == 'f1':
            scorer = make_scorer(metrics.f1_score)
        elif scorer == 'auc':
            scorer = make_scorer(metrics.roc_auc_score)
        for rnd_state in [41, 42, 43]:
            kf = KFold(n_splits=5, shuffle=True, random_state=rnd_state)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring=scorer)
            print(f"Cross-validation scores for random state {rnd_state}: {cv_scores}")
            print(f"Mean cross-validation score for random state {rnd_state}: {cv_scores.mean()}")
            cv_scores_list.append(cv_scores.mean())
        return cv_scores_list

    def get_cv_score_reg(self, model: object, scorer: str):
        cv_scores_list = []
        if scorer == 'r2':
            scorer = make_scorer(metrics.r2_score)
        elif scorer == 'mae':
            scorer = make_scorer(metrics.mean_absolute_error)
        elif scorer == 'mse':
            scorer = make_scorer(metrics.mean_squared_error)
        for rnd_state in [41, 42, 43]:
            kf = KFold(n_splits=5, shuffle=True, random_state=rnd_state)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring=scorer)
            print(f"Cross-validation scores for random state {rnd_state}: {cv_scores}")
            print(f"Mean cross-validation score for random state {rnd_state}: {cv_scores.mean()}")
            cv_scores_list.append(cv_scores.mean())
        return cv_scores_list

    def get_all_cv_scores(self, model: object):
        self.cv_scores_clf = ["matthews_corrcoef", "accuracy", "precision", "recall", "f1", "auc"]
        self.cv_scores_reg = ["r2", "mae", "mse"]
        results = {}
        if self.classification:
            for scorer in self.cv_scores_clf:
                results[scorer] = list(self.get_cv_score_clf(model, scorer))
        else:
            for scorer in self.cv_scores_reg:
                results[scorer] = list(self.get_cv_score_reg(model, scorer))
        # Create DataFrame with model_name as row index and metrics as columns
        # Each cell contains the list of 3 CV scores
        # Use from_dict with orient='index' to create row-wise, then transpose
        df = pd.DataFrame.from_dict({self.model_name: results}, orient='index')
        return df
    
    def get_leave_one_out_score(self, model: object):
        y_true_list = []
        y_pred_list = []
        if not self.classification:
            loo = LeaveOneOut()
            for train_index, test_index in loo.split(self.X_train):
                X_train, X_test = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
                y_train, y_test = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_true_list.append(y_test.values[0])
                y_pred_list.append(y_pred)

            # print(f"r2_score: {metrics.r2_score(y_true_list, y_pred_list)}")
            # print(f"mean_absolute_error: {metrics.mean_absolute_error(y_true_list, y_pred_list)}")
            # print(f"mean_squared_error: {metrics.mean_squared_error(y_true_list, y_pred_list)}")
            return pd.DataFrame(data={"r2_score": metrics.r2_score(y_true_list, y_pred_list), 
            "mean_absolute_error": metrics.mean_absolute_error(y_true_list, y_pred_list), 
            "mean_squared_error": metrics.mean_squared_error(y_true_list, y_pred_list)},
            index=[self.model_name])
            
        else:
            # raise error that classification is not supported for leave one out
            raise ValueError("Classification is not supported for leave one out")
        
    
    def get_mae_score(self, model: object):
        mae_train = metrics.mean_absolute_error(self.y_train, model.predict(self.X_train))
        mae_test = metrics.mean_absolute_error(self.y_test, model.predict(self.X_test))
        return mae_train, mae_test

    def plot_williams_plot(self, model: object, save_path: str = None):
        """
        Plot Williams plot (Domain of Applicability) for regression models.
        X-axis: Leverage values
        Y-axis: Standardized residuals
        """
        if self.classification:
            raise ValueError("Williams plot is only available for regression models")

        # Get predictions and residuals for training data
        y_pred_train = model.predict(self.X_train)
        residuals_train = self.y_train.values.ravel() - y_pred_train

        # Calculate standardized residuals using training data standard deviation
        residual_std = np.std(residuals_train, ddof=1)
        standardized_residuals_train = residuals_train / residual_std

        # Calculate leverage for training data
        # For leverage calculation, we need the hat matrix
        # H = X(X^T X)^(-1) X^T
        X_train = self.X_train.values
        X_design_train = np.column_stack([np.ones(X_train.shape[0]), X_train])  # Add intercept column

        try:
            # Calculate (X^T X)^(-1) using training data
            XtX_inv = np.linalg.inv(X_design_train.T @ X_design_train)
            # Calculate hat matrix diagonal elements (leverage values) for training data
            leverage_train = np.diag(X_design_train @ XtX_inv @ X_design_train.T)

            # Calculate leverage for test data using the same transformation
            X_test = self.X_test.values
            X_design_test = np.column_stack([np.ones(X_test.shape[0]), X_test])
            leverage_test = np.diag(X_design_test @ XtX_inv @ X_design_test.T)

        except np.linalg.LinAlgError:
            print("Warning: Could not calculate leverage due to singular matrix. Using approximation.")
            # Fallback: approximate leverage as 1/n for all points
            leverage_train = np.ones(len(X_train)) / len(X_train)
            leverage_test = np.ones(len(X_test)) / len(X_train)  # Use training n for consistency

        # Get predictions and residuals for test data
        y_pred_test = model.predict(self.X_test)
        residuals_test = self.y_test.values.ravel() - y_pred_test
        standardized_residuals_test = residuals_test / residual_std

        # Calculate warning limits
        n = len(self.X_train)  # number of observations
        p = len(self.descriptor_list) + 1  # number of parameters (features + intercept)
        leverage_threshold = 3 * p / n  # leverage warning limit

        residual_threshold = 3  # standardized residual warning limit (±3)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the training data points (color-blind friendly blue)
        ax.scatter(leverage_train, standardized_residuals_train, alpha=0.7, color='#0072B2', edgecolors='black', s=80, label='Training Data')

        # Plot the test data points (color-blind friendly yellow/orange)
        ax.scatter(leverage_test, standardized_residuals_test, alpha=0.7, color='#E69F00', edgecolors='black', s=80, label='Test Data')

        # Add warning lines (without legend labels)
        ax.axhline(y=residual_threshold, color='black', linestyle='--', alpha=0.7)
        ax.axhline(y=-residual_threshold, color='black', linestyle='--', alpha=0.7)
        ax.axvline(x=leverage_threshold, color='black', linestyle='--', alpha=0.7)

        # Add labels and title with hat value in x-axis label
        ax.set_xlabel(f'Leverage (h* = {leverage_threshold:.3f})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Standardized Residuals', fontsize=14, fontweight='bold')
        # ax.set_title(f'Williams Plot - Domain of Applicability\n{self.model_name}', fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(False)

        # Add legend for data sets only and put it in the upper right corner
        ax.legend(loc='upper right')

        # Set axis limits with some padding
        all_leverage = np.concatenate([leverage_train, leverage_test])
        all_residuals = np.concatenate([standardized_residuals_train, standardized_residuals_test])
        ax.set_xlim(left=0, right=max(all_leverage.max(), leverage_threshold) * 1.1)
        y_max = max(abs(all_residuals.min()), abs(all_residuals.max()), residual_threshold) * 1.1
        ax.set_ylim(-y_max, y_max)

        plt.tight_layout()

        # Save the plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Williams plot saved to: {save_path}")

        return fig, ax

    def plot_scatter_plot(self, model: object, save_path: str = None):
        """
        Plot scatter plot of predicted vs true values for regression models.
        X-axis: Predicted values
        Y-axis: True values
        """
        if self.classification:
            raise ValueError("Scatter plot is only available for regression models")

        # Get predictions for training and test data
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Get true values
        y_true_train = self.y_train.values.ravel()
        y_true_test = self.y_test.values.ravel()
        # print range of the true values
        print(f"range of the true training values: {np.min(y_true_train)} to {np.max(y_true_train)}")
        print(f"range of the true test values: {np.min(y_true_test)} to {np.max(y_true_test)}")

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the training data points (color-blind friendly blue)
        # Note: X-axis is Experimental values, Y-axis is Predicted values
        ax.scatter(y_true_train, y_pred_train, color='#0072B2', alpha=0.7, s=80, edgecolors='black', label='Training Data')

        # Plot the test data points (color-blind friendly yellow/orange)
        ax.scatter(y_true_test, y_pred_test, color='#E69F00', alpha=0.7, s=80, edgecolors='black', label='Test Data')

        # Calculate the range for the diagonal line using all values (predicted and true, train and test)
        all_values = np.concatenate([y_pred_train, y_pred_test, y_true_train, y_true_test])
        min_line = np.min(all_values)
        max_line = np.max(all_values)
        
        # Add padding to ensure the line covers the entire diagonal
        padding = (max_line - min_line) * 0.05
        min_line -= padding
        max_line += padding
        
        # Add a diagonal line (y=x) that covers the entire range
        ax.plot([min_line, max_line], [min_line, max_line], color='black', linewidth=2, linestyle='-')

        # Add labels and title
        ax.set_xlabel('Experimental values', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted values', fontsize=14, fontweight='bold')
        # ax.set_title(f'Scatter Plot of Predicted vs True\n{self.model_name}', fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(False)

        # Add legend
        ax.legend(loc='upper left')

        # Set equal aspect ratio and limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_line, max_line)
        ax.set_ylim(min_line, max_line)

        plt.tight_layout()

        # Save the plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to: {save_path}")

        return fig, ax

    def plot_shap_beeswarm(self, model: object, save_path: str = None, max_display: int = 20, sample_size: int = None):
        """
        Plot SHAP beeswarm plot for regression models.
        Shows the distribution of SHAP values for each feature.
        
        Parameters:
        -----------
        model : object
            The trained regression model
        save_path : str, optional
            Path to save the plot. If None, plot is not saved.
        max_display : int, default=20
            Maximum number of features to display in the plot
        sample_size : int, optional
            Number of samples to use for SHAP value calculation. 
            If None, uses all training data (may be slow for large datasets).
        """
        if self.classification:
            raise ValueError("SHAP beeswarm plot is only available for regression models")

        # Create SHAP explainer
        # Use Explainer which automatically selects the best explainer for the model
        explainer = shap.Explainer(model, self.X_train)
        
        # Calculate SHAP values
        # If sample_size is provided, use a subset of training data
        if sample_size is not None and sample_size < len(self.X_train):
            X_shap = self.X_train.sample(n=min(sample_size, len(self.X_train)), random_state=42)
        else:
            X_shap = self.X_train
        
        print(f"Calculating SHAP values for {len(X_shap)} samples...")
        shap_values = explainer(X_shap)
        
        # Create custom green-to-red colormap
        colors = ['green', 'red']
        cmap = LinearSegmentedColormap.from_list('', colors)
        
        # Create beeswarm plot using summary_plot with cmap parameter
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, max_display=max_display, cmap=cmap, show=False)
        
        # Get current figure and axis for saving
        fig = plt.gcf()
        ax = plt.gca()
        
        plt.tight_layout()
        
        # Save the plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP beeswarm plot saved to: {save_path}")
        
        return fig, ax
       

    def run_each_model(self, leave_one_out: bool = False):
        
        model_handler = ModelHandler(folder_dir=self.folder_dir, classification=self.classification)
        model_handler.load_model()
        model_handler.load_data()
        if leave_one_out:
            return model_handler.get_leave_one_out_score(model_handler.model)
        else:
            return model_handler.get_all_cv_scores(model_handler.model)

# run the code
# uncomment where necessary 
if __name__ == "__main__":
    # # get all folders in the classification folder
    # results_df = pd.DataFrame()
    # classification_folders = glob.glob('./classification/*')
    # for folder in classification_folders:
    #     print(folder)
    #     model_handler = ModelHandler(folder_dir=folder)
    #     results = model_handler.run_each_model(leave_one_out=False)
    #     # print(results)
    #     results_df = pd.concat([results_df, results])
    # results_df.to_csv('classification_results.csv', index=True)

    # get all folders in the regression folder
    results_df = pd.DataFrame()
    regression_folders = glob.glob('./regression/*')
    # regression_folders = ['./regression/vfischeri']
    for folder in regression_folders:
        print(folder)
        model_handler = ModelHandler(folder_dir=folder, classification=False)
        model_handler.load_model()
        model_handler.load_data()
        # mae_train, mae_test = model_handler.get_mae_score(model_handler.model)
        # print(f"mae_train: {mae_train}, mae_test: {mae_test}")
        # results = model_handler.run_each_model(leave_one_out=False)
        # results_df = pd.concat([results_df, results])
        # model_handler.plot_shap_beeswarm(model_handler.model, save_path=f'{folder}/shap_beeswarm.png')

        # Williams plot and scatter plot for each model
        model_handler.plot_williams_plot(model_handler.model, save_path=f'{folder}/williams_plot.png')
        model_handler.plot_scatter_plot(model_handler.model, save_path=f'{folder}/scatter_plot.png')
    # results_df.to_csv('regression_results.csv', index=True)
