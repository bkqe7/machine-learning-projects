import helper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path',dest = 'checkpoint_path',action = 'store', default = 'checkpoint.pth')
parser.add_argument('--image_path',dest = 'image_path',action = 'store', default = './flowers/test/1/image_06743.jpg')
parser.add_argument('--topk',dest = 'topk',action ='store',default =5,type = int)
parser.add_argument('--json_file',dest = 'json_file',action ='store',default = 'cat_to_name.json')

args = parser.parse_args()

cat_to_name = helper.load_json(args.json_file)
checkpoint_model = helper.load_checkpoint(args.checkpoint_path)

top_p,labels = helper.predict(args.image_path,checkpoint_model,cat_to_name,args.topk)

print(top_p,labels)