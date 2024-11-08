import argparse
from model.style_vgg import *
import numpy as np

def get_args():
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description="style transfer by vggNet usage")
    
    # 명령줄 인자 추가 (약어 포함)
    parser.add_argument('-s', '--style_path', type=str, required=True, help="style image path")
    parser.add_argument('-c', '--content_path', type=str, required=True, help="content image path")
    parser.add_argument('-o', '--output_path', type=str, required=True, help="output image path")
    parser.add_argument('-d', '--device', type=str, default="cpu", help="device: cpu or cuda")
    
    # 인자 파싱
    return parser.parse_args()

def save_img(generated_img, output_path):
    gen_img = postprocess(generated_img.cpu()).data.numpy()
    if output_path[-3:] != "jpg" and output_path[-3:] != "png":
        print("type Error")
        return

    int_gen_img = np.clip(gen_img * 255, 0, 255).astype(np.uint8)
    try:
        img = Image.fromarray(int_gen_img)
        img.save(output_path)
    except Exception as e:
        print("image save fail:", e)

def main(args):
    print("style path: ", args.style_path)
    print("content path: ", args.content_path)
    print("output path: ", args.output_path)
    print("device: ", args.device)
    
    content_block = 4
    
    content_img = preprocess(args.content_path).to(args.device)
    style_img = preprocess(args.style_path).to(args.device)
    generated_img = content_img.clone().requires_grad_().to(args.device)
    
    vgg_feature = make_net(args.device)
    
    style_target = list(GramMatrix().to(args.device)(feature_map) for feature_map in vgg_feature(style_img))
    content_target = vgg_feature(content_img)[content_block-1]
    style_weight = [ 1/(n**2)   for n in [64,128,256,512,512] ]
    
    optimizer = torch.optim.Adam([generated_img], lr=0.01)
    
    for epoch in range(500):
        alpha = 10
        
        optimizer.zero_grad()
        out = vgg_feature(generated_img)
        
        style_loss = [ GramMSELoss().to(args.device)(out[i], style_target[i]) * style_weight[i] for i in range(5) ]
        content_loss = nn.MSELoss().to(args.device)(out[content_block - 1], content_target)
        
        total_loss = alpha * sum(style_loss) + torch.mean(content_loss)
        
        total_loss.backward()
        if epoch % 100 == 0:
            print(f"epoch: {epoch}, total loss: {total_loss.item()}")
            
        optimizer.step()
    
    save_img(generated_img, args.output_path)
    
        
        

if __name__ == "__main__":
    args = get_args()
    
    main(args)