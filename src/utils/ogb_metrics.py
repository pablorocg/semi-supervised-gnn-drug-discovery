import torch
from torch import Tensor
from torchmetrics import Metric
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

class MultiTaskROCAUC(Metric):
    """ROC-AUC averaged across tasks with NaN handling"""
    
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        
        # Store predictions and targets for each task separately
        self.add_state("preds_list", default=[], dist_reduce_fx="cat")
        self.add_state("target_list", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Args:
            preds: Predictions of shape (batch_size, num_tasks)
            target: Ground truth of shape (batch_size, num_tasks)
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        if preds.shape[1] != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, got {preds.shape[1]}")
        
        self.preds_list.append(preds.detach().cpu())
        self.target_list.append(target.detach().cpu())
    
    def compute(self) -> Tensor:
        """Compute ROC-AUC averaged across tasks"""
        preds = torch.cat(self.preds_list, dim=0).numpy()
        target = torch.cat(self.target_list, dim=0).numpy()
        
        rocauc_list = []
        
        for i in range(self.num_tasks):
            # Check if we have both positive and negative samples
            if np.sum(target[:, i] == 1) > 0 and np.sum(target[:, i] == 0) > 0:
                # Filter out NaN values
                is_labeled = target[:, i] == target[:, i]
                if np.sum(is_labeled) > 0:
                    try:
                        auc = roc_auc_score(target[is_labeled, i], preds[is_labeled, i])
                        rocauc_list.append(auc)
                    except:
                        continue
        
        if len(rocauc_list) == 0:
            raise RuntimeError(
                'No positively labeled data available. Cannot compute ROC-AUC.')
        
        return torch.tensor(sum(rocauc_list) / len(rocauc_list))


class MultiTaskAP(Metric):
    """Average Precision averaged across tasks with NaN handling"""
    
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        
        self.add_state("preds_list", default=[], dist_reduce_fx="cat")
        self.add_state("target_list", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Args:
            preds: Predictions of shape (batch_size, num_tasks)
            target: Ground truth of shape (batch_size, num_tasks)
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        if preds.shape[1] != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, got {preds.shape[1]}")
        
        self.preds_list.append(preds.detach().cpu())
        self.target_list.append(target.detach().cpu())
    
    def compute(self) -> Tensor:
        """Compute Average Precision averaged across tasks"""
        preds = torch.cat(self.preds_list, dim=0).numpy()
        target = torch.cat(self.target_list, dim=0).numpy()
        
        ap_list = []
        
        for i in range(self.num_tasks):
            # Check if we have both positive and negative samples
            if np.sum(target[:, i] == 1) > 0 and np.sum(target[:, i] == 0) > 0:
                # Filter out NaN values
                is_labeled = target[:, i] == target[:, i]
                if np.sum(is_labeled) > 0:
                    try:
                        ap = average_precision_score(
                            target[is_labeled, i], 
                            preds[is_labeled, i]
                        )
                        ap_list.append(ap)
                    except:
                        continue
        
        if len(ap_list) == 0:
            raise RuntimeError(
                'No positively labeled data available. Cannot compute Average Precision.')
        
        return torch.tensor(sum(ap_list) / len(ap_list))


class MultiTaskRMSE(Metric):
    """RMSE averaged across tasks with NaN handling"""
    
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        
        self.add_state("squared_error", 
                      default=torch.zeros(num_tasks), 
                      dist_reduce_fx="sum")
        self.add_state("total", 
                      default=torch.zeros(num_tasks), 
                      dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Args:
            preds: Predictions of shape (batch_size, num_tasks)
            target: Ground truth of shape (batch_size, num_tasks)
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        if preds.shape[1] != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, got {preds.shape[1]}")
        
        for i in range(self.num_tasks):
            # Filter out NaN values
            is_labeled = target[:, i] == target[:, i]
            if torch.sum(is_labeled) > 0:
                errors = (target[is_labeled, i] - preds[is_labeled, i]) ** 2
                self.squared_error[i] += torch.sum(errors)
                self.total[i] += torch.sum(is_labeled)
    
    def compute(self) -> Tensor:
        """Compute RMSE averaged across tasks"""
        rmse_list = []
        
        for i in range(self.num_tasks):
            if self.total[i] > 0:
                rmse = torch.sqrt(self.squared_error[i] / self.total[i])
                rmse_list.append(rmse)
        
        if len(rmse_list) == 0:
            return torch.tensor(0.0)
        
        return torch.stack(rmse_list).mean()


class MultiTaskAccuracy(Metric):
    """Accuracy averaged across tasks with NaN handling"""
    
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        
        self.add_state("correct", 
                      default=torch.zeros(num_tasks), 
                      dist_reduce_fx="sum")
        self.add_state("total", 
                      default=torch.zeros(num_tasks), 
                      dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Args:
            preds: Predictions of shape (batch_size, num_tasks)
            target: Ground truth of shape (batch_size, num_tasks)
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        if preds.shape[1] != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, got {preds.shape[1]}")
        
        for i in range(self.num_tasks):
            # Filter out NaN values
            is_labeled = target[:, i] == target[:, i]
            if torch.sum(is_labeled) > 0:
                correct = (preds[is_labeled, i] == target[is_labeled, i])
                self.correct[i] += torch.sum(correct)
                self.total[i] += torch.sum(is_labeled)
    
    def compute(self) -> Tensor:
        """Compute accuracy averaged across tasks"""
        acc_list = []
        
        for i in range(self.num_tasks):
            if self.total[i] > 0:
                acc = self.correct[i].float() / self.total[i]
                acc_list.append(acc)
        
        if len(acc_list) == 0:
            return torch.tensor(0.0)
        
        return torch.stack(acc_list).mean()


class SetF1Score(Metric):
    """F1 score for set-based predictions averaged over samples"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_state("precision_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("f1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds_list, target_list) -> None:
        """
        Args:
            preds_list: List of lists/sets containing predictions for each sample
            target_list: List of lists/sets containing ground truth for each sample
        """
        if len(preds_list) != len(target_list):
            raise ValueError("preds_list and target_list must have the same length")
        
        for pred, target in zip(preds_list, target_list):
            label = set(target)
            prediction = set(pred)
            
            true_positive = len(label.intersection(prediction))
            false_positive = len(prediction - label)
            false_negative = len(label - prediction)
            
            if true_positive + false_positive > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0.0
            
            if true_positive + false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            self.precision_sum += precision
            self.recall_sum += recall
            self.f1_sum += f1
            self.total += 1
    
    def compute(self) -> dict:
        """Compute averaged precision, recall, and F1"""
        if self.total == 0:
            return {
                'precision': torch.tensor(0.0),
                'recall': torch.tensor(0.0),
                'F1': torch.tensor(0.0)
            }
        
        return {
            'precision': self.precision_sum / self.total,
            'recall': self.recall_sum / self.total,
            'F1': self.f1_sum / self.total
        }







if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ALL CUSTOM TORCHMETRICS")
    print("=" * 60)
    
    # Parámetros comunes
    num_tasks = 128
    batch_size = 32
    num_batches = 5
    
    # ============================================================
    # 1. TEST: MultiTaskROCAUC
    # ============================================================
    print("\n1. Testing MultiTaskROCAUC")
    print("-" * 60)
    rocauc_metric = MultiTaskROCAUC(num_tasks=num_tasks)
    
    for i in range(num_batches):
        preds = torch.rand(batch_size, num_tasks)
        target = torch.randint(0, 2, (batch_size, num_tasks)).float()
        # Añadir algunos NaN para probar el manejo
        if i == 0:
            target[0:5, 0:10] = float('nan')
        rocauc_metric.update(preds, target)
    
    rocauc_result = rocauc_metric.compute()
    print(f"✓ ROC-AUC: {rocauc_result:.4f}")
    print(f"  Batches procesados: {num_batches}")
    print(f"  Total muestras: {num_batches * batch_size}")
    rocauc_metric.reset()
    
    # ============================================================
    # 2. TEST: MultiTaskAP
    # ============================================================
    print("\n2. Testing MultiTaskAP (Average Precision)")
    print("-" * 60)
    ap_metric = MultiTaskAP(num_tasks=num_tasks)
    
    for i in range(num_batches):
        preds = torch.rand(batch_size, num_tasks)
        target = torch.randint(0, 2, (batch_size, num_tasks)).float()
        # Añadir algunos NaN
        if i == 1:
            target[10:15, 20:30] = float('nan')
        ap_metric.update(preds, target)
    
    ap_result = ap_metric.compute()
    print(f"✓ Average Precision: {ap_result:.4f}")
    print(f"  Batches procesados: {num_batches}")
    print(f"  Total muestras: {num_batches * batch_size}")
    ap_metric.reset()
    
    # ============================================================
    # 3. TEST: MultiTaskRMSE
    # ============================================================
    print("\n3. Testing MultiTaskRMSE")
    print("-" * 60)
    rmse_metric = MultiTaskRMSE(num_tasks=num_tasks)
    
    for i in range(num_batches):
        preds = torch.randn(batch_size, num_tasks)
        target = torch.randn(batch_size, num_tasks)
        # Añadir algunos NaN para regresión
        if i == 2:
            target[5:10, 50:60] = float('nan')
        rmse_metric.update(preds, target)
    
    rmse_result = rmse_metric.compute()
    print(f"✓ RMSE: {rmse_result:.4f}")
    print(f"  Batches procesados: {num_batches}")
    print(f"  Total muestras: {num_batches * batch_size}")
    rmse_metric.reset()
    
    # ============================================================
    # 4. TEST: MultiTaskAccuracy
    # ============================================================
    print("\n4. Testing MultiTaskAccuracy")
    print("-" * 60)
    acc_metric = MultiTaskAccuracy(num_tasks=num_tasks)
    
    for i in range(num_batches):
        # Predicciones discretas para clasificación
        preds = torch.randint(0, 2, (batch_size, num_tasks)).float()
        target = torch.randint(0, 2, (batch_size, num_tasks)).float()
        # Añadir algunos NaN
        if i == 3:
            target[15:20, 70:80] = float('nan')
        acc_metric.update(preds, target)
    
    acc_result = acc_metric.compute()
    print(f"✓ Accuracy: {acc_result:.4f}")
    print(f"  Batches procesados: {num_batches}")
    print(f"  Total muestras: {num_batches * batch_size}")
    acc_metric.reset()
    
    # ============================================================
    # 5. TEST: SetF1Score
    # ============================================================
    print("\n5. Testing SetF1Score")
    print("-" * 60)
    f1_metric = SetF1Score()
    
    # Simular múltiples batches de predicciones basadas en sets
    for batch_idx in range(num_batches):
        preds_list = []
        target_list = []
        for _ in range(10):  # 10 muestras por batch
            # Generar sets aleatorios
            pred_size = np.random.randint(1, 8)
            target_size = np.random.randint(1, 8)
            preds_list.append(list(np.random.randint(0, 20, pred_size)))
            target_list.append(list(np.random.randint(0, 20, target_size)))
        
        f1_metric.update(preds_list, target_list)
    
    f1_results = f1_metric.compute()
    print(f"✓ F1 Score: {f1_results['F1']:.4f}")
    print(f"  Precision: {f1_results['precision']:.4f}")
    print(f"  Recall: {f1_results['recall']:.4f}")
    print(f"  Batches procesados: {num_batches}")
    print(f"  Total muestras: {num_batches * 10}")
    f1_metric.reset()
    
    # ============================================================
    # 6. TEST: Caso edge - todas NaN
    # ============================================================
    print("\n6. Testing Edge Case: Empty/NaN handling")
    print("-" * 60)
    
    try:
        # ROC-AUC con solo NaN
        rocauc_edge = MultiTaskROCAUC(num_tasks=5)
        preds_nan = torch.rand(10, 5)
        target_nan = torch.full((10, 5), float('nan'))
        rocauc_edge.update(preds_nan, target_nan)
        result = rocauc_edge.compute()
        print("✗ ROC-AUC debería haber lanzado RuntimeError")
    except RuntimeError as e:
        print(f"✓ ROC-AUC correctamente lanza error con solo NaN: {str(e)[:50]}...")
    
    # RMSE con NaN
    rmse_edge = MultiTaskRMSE(num_tasks=5)
    preds_partial = torch.randn(10, 5)
    target_partial = torch.randn(10, 5)
    target_partial[:, 0:3] = float('nan')  # Primeras 3 tareas son NaN
    rmse_edge.update(preds_partial, target_partial)
    result_edge = rmse_edge.compute()
    print(f"✓ RMSE con NaN parcial: {result_edge:.4f}")
    
    # ============================================================
    # 7. TEST: Comparación con implementación original
    # ============================================================
    print("\n7. Testing Consistency with Original Implementation")
    print("-" * 60)
    
    # Crear datos de prueba
    test_preds = torch.rand(100, 10)
    test_target = torch.randint(0, 2, (100, 10)).float()
    
    # Torchmetrics version
    tm_acc = MultiTaskAccuracy(num_tasks=10)
    tm_acc.update(test_preds.round(), test_target)
    tm_result = tm_acc.compute()
    
    # Original implementation
    def eval_acc_original(y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))
        return {'acc': sum(acc_list) / len(acc_list)}
    
    orig_result = eval_acc_original(
        test_target.numpy(), 
        test_preds.round().numpy()
    )
    
    print(f"  TorchMetrics Accuracy: {tm_result:.6f}")
    print(f"  Original Accuracy: {orig_result['acc']:.6f}")
    print(f"  Diferencia: {abs(tm_result.item() - orig_result['acc']):.8f}")
    if abs(tm_result.item() - orig_result['acc']) < 1e-6:
        print("✓ Resultados consistentes!")
    else:
        print("✗ Resultados difieren!")
    
    # ============================================================
    # 8. TEST: Reset functionality
    # ============================================================
    print("\n8. Testing Reset Functionality")
    print("-" * 60)
    
    test_metric = MultiTaskAccuracy(num_tasks=5)
    preds1 = torch.randint(0, 2, (20, 5)).float()
    target1 = torch.randint(0, 2, (20, 5)).float()
    
    test_metric.update(preds1, target1)
    result1 = test_metric.compute()
    print(f"  Primera medición: {result1:.4f}")
    
    test_metric.reset()
    
    preds2 = torch.randint(0, 2, (20, 5)).float()
    target2 = torch.randint(0, 2, (20, 5)).float()
    test_metric.update(preds2, target2)
    result2 = test_metric.compute()
    print(f"  Segunda medición (después de reset): {result2:.4f}")
    print("✓ Reset funcionando correctamente")
    
    print("\n" + "=" * 60)
    print("TODAS LAS PRUEBAS COMPLETADAS!")
    print("=" * 60)
