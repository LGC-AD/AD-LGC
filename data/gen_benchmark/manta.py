import os
import json


class MANTASolver(object):
    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test-pixel']
        self.CLSNAMES = os.listdir(root)

    def run(self):
        info = {phase: {} for phase in self.phases}
        for cls_name in self.CLSNAMES:
            if 'tar.gz' in cls_name:
                continue
            cls_dir = f'{self.root}/{cls_name}'
            for phase in self.phases:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    if 'mask' in specie:
                        continue
                    is_abnormal = True if specie not in ['good'] else False
                    img_dir = f'{cls_dir}/{phase}/{specie}/'
                    mask_dir = f'{cls_dir}/{phase}/mask'
                    img_names = os.listdir(img_dir)
                    mask_names = os.listdir(mask_dir) if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{img_dir.replace(cls_dir, cls_name)}/{img_name}',
                            mask_path=f'{mask_dir.replace(cls_dir, cls_name)}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print(f'MANTA meta file saved to {self.meta_path}')


if __name__ == '__main__':
    runner = MANTASolver(root='/mnt/lpai-dione/ssai/cvg/team/share_datasets/Grain/MANTA/groceries')
    # runner.run()
    # runner = MVTecSolver(root='data/mvtec3d', is2D=False)
    runner.run()
