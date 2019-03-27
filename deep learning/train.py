import argparse
import helper

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',nargs = '?',type = str, default = './flowers/')
parser.add_argument('--gpu',dest = 'gpu',action = 'store_true',default = False)
parser.add_argument('--save_dir',dest = 'save_dir',action = 'store',default = './checkpoint.pth')
parser.add_argument('--arch',dest = 'arch',action = 'store',default ='vgg16')
parser.add_argument('--learning_rate',dest ='learning_rate',action = 'store',default = 0.001,type = float)
parser.add_argument('--hidden_units',dest = 'hidden_units',action = 'store',default = 1024, type = int )
parser.add_argument('--epochs',dest = 'epochs',action = 'store',default = 20,type = int)

args = parser.parse_args()

# load data
train_data,trainloader, testloader,validloader = helper.load_data()

# build model
print(args.gpu)
print(args.arch)
print(type(args.hidden_units))
print(type(args.learning_rate))
model,device,criterion,optimizer = helper.build_model(args.gpu,args.arch,args.hidden_units,args.learning_rate)

# train model
helper.train_model(args.epochs,trainloader,validloader,model,device,criterion,optimizer)

# save the trained model
helper.save_checkpoint(model,args.epochs,args.arch,optimizer,train_data)

