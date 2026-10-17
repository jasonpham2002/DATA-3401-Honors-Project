import numpy as np

class LogisticRegression:
    """
    Binary Logistic Regression classifier implemented from scratch using NumPy.
    Optimizes weights via batch gradient descent with binary cross-entropy loss.
    Supports balanced class weighting to prevent majority-class bias in trending markets.
    Includes comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
    """
    def __init__(self, learning_rate=0.05, n_epochs=3000, class_weight='balanced'):
        self.lr = learning_rate
        self.epochs = n_epochs
        self.class_weight = class_weight
        self.w = None
        self.sample_weights = None
        self._w1 = None
        self._w0 = None
        self.cost_history = []

    def sigmoid_(self, z):
        """Numerically stable sigmoid function with bounds clipping."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _prepare_xy_weights(self, X, y):
        """Ensures X has bias column matching self.w, y is 2D column vector, and sample weights align."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if self.w is not None and X.shape[1] == self.w.shape[0] - 1:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        weights = None
        if self.sample_weights is not None:
            if self.sample_weights.shape == y.shape:
                weights = self.sample_weights
            elif self._w1 is not None and self._w0 is not None:
                weights = np.where(y == 1, self._w1, self._w0)
        return X, y, weights

    def compute_cost(self, X, y):
        """
        Computes (weighted) Binary Cross-Entropy (Log Loss):
        J(w) = -1/m * sum( w_i * [y * log(y_hat) + (1 - y) * log(1 - y_hat)] )
        """
        X, y, weights = self._prepare_xy_weights(X, y)
        m = X.shape[0]
        y_hat = self.sigmoid_(X @ self.w)
        y_hat = np.clip(y_hat, 1e-15, 1.0 - 1e-15)
        raw_loss = y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)
        if weights is not None:
            j = (-1.0 / m) * np.sum(weights * raw_loss)
        else:
            j = (-1.0 / m) * np.sum(raw_loss)
        return float(j)

    def compute_gradient(self, X, y):
        """
        Computes gradient of BCE loss with respect to weights:
        dj_dw = 1/m * (X.T @ (w_i * (y_hat - y)))
        """
        X, y, weights = self._prepare_xy_weights(X, y)
        m = X.shape[0]
        y_hat = self.sigmoid_(X @ self.w)
        error = y_hat - y
        if weights is not None:
            error = error * weights
        dj_dw = (1.0 / m) * (X.T @ error)
        return dj_dw

    def gradient_descent(self, X, y):
        """Runs gradient descent optimization and records cost history."""
        self.cost_history = []
        for i in range(self.epochs):
            grad = self.compute_gradient(X, y)
            self.w = self.w - self.lr * grad
            self.cost_history.append(self.compute_cost(X, y))

    def fit(self, X, y):
        """
        Trains model weights on feature matrix X and label vector y.
        Automatically prepends an intercept column of ones to X.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if self.class_weight == 'balanced':
            p1 = np.mean(y == 1)
            p0 = np.mean(y == 0)
            self._w1 = (0.5 / p1) if p1 > 0 else 1.0
            self._w0 = (0.5 / p0) if p0 > 0 else 1.0
            self.sample_weights = np.where(y == 1, self._w1, self._w0)
        else:
            self._w1 = None
            self._w0 = None
            self.sample_weights = None

        m = X.shape[0]
        ones = np.ones((m, 1))
        X_with_bias = np.c_[ones, X]
        n = X_with_bias.shape[1]

        # Initialize weights to zeros (shape: (n, 1))
        self.w = np.zeros((n, 1))
        self.gradient_descent(X_with_bias, y)

    def predict_proba(self, X):
        """Returns predicted probability P(y=1|X)."""
        X = np.asarray(X, dtype=float)
        m = X.shape[0]
        ones = np.ones((m, 1))
        X_with_bias = np.c_[ones, X]
        return self.sigmoid_(X_with_bias @ self.w)

    def predict(self, X, threshold=0.22, smooth_window=3):
        """
        Returns binary regime predictions (0 or 1) based on probability threshold.
        Applies a pure-NumPy rolling smoothing window to filter single-day probability noise.
        """
        probs = self.predict_proba(X).ravel()
        if smooth_window is not None and smooth_window > 1:
            probs = np.array([
                np.mean(probs[max(0, i - smooth_window + 1):i + 1])
                for i in range(len(probs))
            ])
        return (probs >= threshold).astype(int).reshape(-1, 1)

    def accuracy_test(self, y_true, y_predict):
        """Calculates classification accuracy."""
        y_true = np.asarray(y_true).ravel()
        y_predict = np.asarray(y_predict).ravel()
        return float(np.mean(y_true == y_predict)) if len(y_true) > 0 else 0.0

    def confusion_matrix(self, y_true, y_pred):
        """
        Computes 2x2 confusion matrix:
        [[True Negative,  False Positive],
         [False Negative, True Positive ]]
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        return np.array([[tn, fp], [fn, tp]])

    def precision_score(self, y_true, y_pred):
        """Calculates Precision: TP / (TP + FP)."""
        cm = self.confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fp = cm[0, 1]
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def recall_score(self, y_true, y_pred):
        """Calculates Recall (Sensitivity): TP / (TP + FN)."""
        cm = self.confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fn = cm[1, 0]
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def f1_score(self, y_true, y_pred):
        """Calculates F1-Score: 2 * (Precision * Recall) / (Precision + Recall)."""
        p = self.precision_score(y_true, y_pred)
        r = self.recall_score(y_true, y_pred)
        return float((2.0 * p * r) / (p + r)) if (p + r) > 0 else 0.0

    def baseline_accuracy(self, y_true):
        """
        Calculates naive baseline accuracy of always predicting the majority class (Bullish=1).
        """
        y_true = np.asarray(y_true).ravel()
        return float(np.mean(y_true == 1)) if len(y_true) > 0 else 0.0

    def evaluate(self, y_true, y_pred):
        """Returns comprehensive dictionary of all evaluation metrics."""
        return {
            "Accuracy": self.accuracy_test(y_true, y_pred),
            "Precision": self.precision_score(y_true, y_pred),
            "Recall": self.recall_score(y_true, y_pred),
            "F1-Score": self.f1_score(y_true, y_pred),
            "Baseline (Always Bullish)": self.baseline_accuracy(y_true),
            "Confusion Matrix": self.confusion_matrix(y_true, y_pred)
        }
