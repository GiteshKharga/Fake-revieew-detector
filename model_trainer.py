import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import calculate_metrics, save_metrics_to_csv
from tqdm import tqdm
import time

class ModelTrainer:
    def __init__(self):
        print("\nInitializing models with optimized configurations...")
        # Define model configurations with optimized hyperparameters
        self.models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=8,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            ),
            'Linear SVC': LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
        }
        self.results = {}
        
    def optimize_model(self, model, X_train, y_train, param_grid):
        """Optimize model hyperparameters using GridSearchCV"""
        print(f"\nOptimizing {model.__class__.__name__}...")
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models with optimization"""
        metrics_data = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'F1 Score': [],
            'CV Score': []
        }
        
        # Define parameter grids for optimization
        param_grids = {
            'Decision Tree': {
                'max_depth': [8, 10, 12],
                'min_samples_split': [5, 8, 10],
                'min_samples_leaf': [2, 4, 6]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8],
                'subsample': [0.7, 0.8, 0.9]
            },
            'Linear SVC': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            },
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12],
                'min_samples_split': [5, 8, 10]
            }
        }
        
        # Create progress bar for model training
        model_progress = tqdm(self.models.items(), desc="Training models", total=len(self.models))
        
        for name, model in model_progress:
            model_progress.set_description(f"Training {name}")
            
            # Optimize model hyperparameters
            optimized_model = self.optimize_model(
                model, X_train, y_train, param_grids[name]
            )
            
            # Train the optimized model
            print(f"\nTraining {name}...")
            optimized_model.fit(X_train, y_train)
            
            # Make predictions
            print("Making predictions...")
            y_pred = optimized_model.predict(X_test)
            
            # Calculate metrics
            print("Calculating metrics...")
            metrics = calculate_metrics(y_test, y_pred)
            cv_scores = cross_val_score(optimized_model, X_train, y_train, cv=5)
            
            # Store results
            self.results[name] = {
                'model': optimized_model,
                'metrics': metrics,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Store metrics for CSV
            metrics_data['Model'].append(name)
            metrics_data['Accuracy'].append(metrics['accuracy'])
            metrics_data['Precision'].append(metrics['precision'])
            metrics_data['F1 Score'].append(metrics['f1'])
            metrics_data['CV Score'].append(cv_scores.mean())
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Create and evaluate ensemble model with hard voting
        print("\nTraining ensemble model...")
        ensemble = VotingClassifier(
            estimators=[(name, results['model']) 
                       for name, results in self.results.items()],
            voting='hard'  # Changed from 'soft' to 'hard' voting
        )
        
        print("Fitting ensemble model...")
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        print("Evaluating ensemble model...")
        y_pred_ensemble = ensemble.predict(X_test)
        metrics_ensemble = calculate_metrics(y_test, y_pred_ensemble)
        cv_scores_ensemble = cross_val_score(ensemble, X_train, y_train, cv=5)
        
        # Store ensemble results
        self.results['Ensemble'] = {
            'model': ensemble,
            'metrics': metrics_ensemble,
            'cv_mean': cv_scores_ensemble.mean(),
            'cv_std': cv_scores_ensemble.std()
        }
        
        # Add ensemble metrics to CSV data
        metrics_data['Model'].append('Ensemble')
        metrics_data['Accuracy'].append(metrics_ensemble['accuracy'])
        metrics_data['Precision'].append(metrics_ensemble['precision'])
        metrics_data['F1 Score'].append(metrics_ensemble['f1'])
        metrics_data['CV Score'].append(cv_scores_ensemble.mean())
        
        # Save metrics to CSV
        print("\nSaving metrics to CSV...")
        save_metrics_to_csv(metrics_data)
    
    def find_best_model(self):
        """Find the best performing model based on F1 score"""
        print("\nFinding best performing model...")
        best_model_name = max(self.results.items(), 
                            key=lambda x: x[1]['metrics']['f1'])[0]
        print(f"Best model: {best_model_name}")
        return best_model_name, self.results[best_model_name]['model']
    
    def save_best_model(self, model, filename='best_model.joblib'):
        """Save the best performing model"""
        print(f"\nSaving best model to {filename}...")
        joblib.dump(model, filename)
        print("Model saved successfully!")
    
    def plot_metrics_comparison(self):
        """Create visualizations comparing model metrics"""
        print("\nCreating performance visualizations...")
        metrics = ['Accuracy', 'Precision', 'F1 Score', 'CV Score']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [results['metrics'].get(metric.lower(), results['cv_mean']) 
                     for results in self.results.values()]
            names = list(self.results.keys())
            
            sns.barplot(x=names, y=values, ax=axes[idx])
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_xticklabels(names, rotation=45)
            axes[idx].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_metrics_comparison.png')
        plt.close()
        print("Visualizations saved as 'model_metrics_comparison.png'")

if __name__ == "__main__":
    print("Starting model training pipeline...")
    from data_processor import DataProcessor
    
    # Prepare data
    processor = DataProcessor("dataset/yelp_academic_dataset_review.json")
    X_train, X_test, y_train, y_test, tfidf, scaler = processor.prepare_data()
    
    # Train and evaluate models
    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Find and save best model
    best_model_name, best_model = trainer.find_best_model()
    trainer.save_best_model(best_model)
    
    # Create visualizations
    trainer.plot_metrics_comparison()
    
    print("\nModel training pipeline completed successfully!") 