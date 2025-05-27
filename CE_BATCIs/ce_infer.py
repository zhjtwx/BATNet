import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('.')
from pro_at_patch import SymmetricalATProcessor
sys.path.append('./CE_BATCIs')
from model.resnet_cim import CEBATCls


class BATInference:
    def __init__(self, model_path, device='cuda', seg_model_file=None, dtypes={'ct': 'float32', 'mask': 'float32'}):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CEBATCls().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.patch_size = 32
        self.dtypes = dtypes
        self.seg_model = None
        if seg_model_file is not None:
            self.seg_model = SymmetricalATProcessor(model_file=seg_model_file)

    def predict_case(self, ct_path, bat_path, at_path, left_patch_path, right_patch_path, left_label_path, right_label_path):
        if os.path.exists(left_patch_path) and  os.path.exists(right_patch_path) \
                and os.path.exists(left_label_path) and os.path.exists(right_label_path):
            left_patch = nib.load(left_patch_path).get_fdata().astype(self.dtypes['ct'])
            right_patch = nib.load(right_patch_path).get_fdata().astype(self.dtypes['ct'])
            left_label = nib.load(left_label_path).get_fdata().astype(self.dtypes['mask'])
            right_label = nib.load(right_label_path).get_fdata().astype(self.dtypes['mask'])
        else:
            # 使用分割模型生成patch
            if self.seg_model is None:
                raise ValueError("Segmentation model is required when cached patches not available")
            # 检查必要的输入文件
            if not os.path.exists(ct_path):
                raise FileNotFoundError(f"CT file not found: {ct_path}")
            at_path = bat_path if os.path.exists(at_path) else None
            bat_path = bat_path if os.path.exists(bat_path) else None

            left_patch, right_patch, left_label, right_label = self.seg_model.process(ct_path, at_path, bat_path)

            # 保存生成的patch供下次使用
            nib.save(nib.Nifti1Image(left_patch, np.eye(4)), left_patch_path)
            nib.save(nib.Nifti1Image(right_patch, np.eye(4)), right_patch_path)
            nib.save(nib.Nifti1Image(left_label, np.eye(4)), left_label_path)
            nib.save(nib.Nifti1Image(right_label, np.eye(4)), right_label_path)

        # 转换为张量
        left_tensor = torch.tensor(left_patch, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1,16,2,32,32,32)
        right_tensor = torch.tensor(right_patch, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            case_prob, patch_preds = self.model(left_tensor, right_tensor)
            case_prob = case_prob.squeeze().item()

            # 转换patch预测结果
            patch_probs = torch.softmax(patch_preds.view(-1, 2), dim=1)[:, 1]  # 获取阳性概率
            patch_probs = patch_probs.cpu().numpy().reshape(16, 2)  # 左右各16个patch

        return {
            'case_prob': case_prob,
            'patch_probs': patch_probs,
            'left_patches': left_patch,
            'right_patches': right_patch
        }

    def visualize_results(self, result, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].bar(['Negative', 'Positive'], [1 - result['case_prob'], result['case_prob']])
        axes[0, 0].set_title(f'Case Level Prediction: {result["case_prob"]:.3f}')
        axes[0, 0].set_ylim(0, 1)

        im = axes[0, 1].imshow(result['patch_probs'], cmap='Reds', vmin=0, vmax=1)
        axes[0, 1].set_title('Patch Prediction Heatmap')
        axes[0, 1].set_xlabel('Left/Right (0:Left, 1:Right)')
        axes[0, 1].set_ylabel('Patch Index')
        plt.colorbar(im, ax=axes[0, 1])

        max_idx = np.unravel_index(np.argmax(result['patch_probs']), result['patch_probs'].shape)
        patch_type = 'left' if max_idx[1] == 0 else 'right'
        sample_patch = result[f'{patch_type}_patches'][max_idx[0]]

        axes[1, 0].imshow(sample_patch[0, 16, :, :], cmap='gray')
        axes[1, 0].set_title(f'Max Prob Patch CT (Prob: {result["patch_probs"][max_idx]:.2f})')

        axes[1, 1].imshow(sample_patch[0, 16, :, :], cmap='gray')
        axes[1, 1].imshow(sample_patch[1, 16, :, :], alpha=0.5, cmap='Reds')
        axes[1, 1].set_title('CT with AT Mask Overlay')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def batch_predict(self, data_dir, output_csv='results.csv'):
        import pandas as pd
        from glob import glob
        import os
        case_dirs = sorted(glob(os.path.join(data_dir, '*/*/*')))

        results = []
        for case_dir in tqdm(case_dirs, desc='Processing Cases'):
            try:
                ct_path = os.path.join(case_dir, 'image.nii.gz')
                bat_path = os.path.join(case_dir, 'brown_fat_mask.nii.gz')
                at_path = os.path.join(case_dir, 'fat_mask.nii.gz')
                left_patch_path = os.path.join(case_dir, 'ct_at_left_patch.nii.gz')
                right_patch_path = os.path.join(case_dir, 'ct_at_right_patch.nii.gz')
                left_label_path = os.path.join(case_dir, 'ct_at_left_label.nii.gz')
                right_label_path = os.path.join(case_dir, 'ct_at_right_label.nii.gz')

                if not os.path.exists(ct_path):
                    continue
                result = self.predict_case(ct_path, bat_path, at_path, left_patch_path,
                                           right_patch_path, left_label_path, right_label_path)

                # self.visualize_results(
                #     result,
                #     save_path=os.path.join(case_dir, 'prediction_visualization.png')
                # )
                results.append({
                    'case_id': os.path.basename(case_dir),
                    'pred_prob': result['case_prob'],
                    'pred_label': int(result['case_prob'] > 0.5),
                    'max_patch_prob': np.max(result['patch_probs'])
                })
            except Exception as e:
                print(f"Error processing {case_dir}: {str(e)}")

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    predictor = BATInference(
        model_path='best_model.pth',
        device='cuda' 
    )
    predictor.batch_predict(
        data_dir='.BATNet/data/cls_data/train',
        output_csv='prediction_results.csv'
    )