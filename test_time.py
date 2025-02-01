import torch  
import time   
from models.test import GGNN
from utils.data.jacquard_data import JacquardDataset
import torch.utils.data


if __name__ == '__main__':
    model = GGNN(in_channels  = 3)                               
    device = torch.device("cuda:0")
    model.to(device)  
    val_dataset = JacquardDataset(file_path = "E:\jacquard_dataset", start=0.9, end=1.0, ds_rotate=False,
                            random_rotate=False, random_zoom=False,
                            include_depth=0, include_rgb=1, include_mask=0, output_size=224)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )
    num_iterations = 1000  # 设置推理迭代次数  

    time_use_all = 0.0 
    idx = 0
    with torch.no_grad():  
        for x, _, _, _, _ in val_data: 
            x = x.to(device)
            start_time = time.time()  
            _, _, _, _ = model(x)
            end_time = time.time()  
            time_use_all = time_use_all + (end_time - start_time) 
            idx += 1
            if idx == 999:
                break
        print("Average time per iteration: ", time_use_all / 1000.0)

# 单位都是秒 jacquard
# net               Average time per iteration      fps        Input          
# mynet             0.006433690071105957            155.47     224x224RGB
# 2020GGCNN         0.0017987513542175293           555.86     300X300RGB
# 2022swim          0.02260865044593811             44.24      224x224RGB
# 2022se_ResUnet    0.0065943353176116945           142.55     224x224RGB
