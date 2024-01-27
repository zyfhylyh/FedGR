from cnn import CNN_MNIST
import json
import torchvision
import torch
import torchvision.transforms as transforms
from socket import *
import asyncio
import argparse
import time
import requests
from threading import Thread
import numpy as np
from torch.utils.data import DataLoader
from fedlab.utils.dataset.sampler import SubsetSampler
from fedlab.utils.functional import evaluate


def random_slicing(dataset, num_clients):
    """Slice a dataset randomly and equally for IID.

    Args：
        dataset (torch.utils.data.Dataset): a dataset for slicing.
        num_clients (int):  the number of client.

    Returns：
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = list(
            np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users

class training_server(Thread):
    def __init__(self,**kwargs):
        Thread.__init__(self)
        self.model=kwargs.get('model')
        self.epoch=kwargs.get('epoch')
        self.batchsize=kwargs.get('batchsize')
        self.lr=kwargs.get('lr')#学习率
        self.partition=kwargs.get('partition')
        self.name=kwargs.get('name')#name是唯一标识符
        self.dataset=kwargs.get('dataset')
        self.rank=0
        self.non_iid=kwargs.get('non_iidv')
        self.aggr_ip=kwargs.get("aggr_ip")
        self.aggr_port= kwargs.get("aggr_port")
        self.task_id=kwargs.get("task_id")
        self.status=-1
        self.currentround=1
        self.traindataloader=kwargs.get("traindataloader")
        self.testdataloader=kwargs.get("testdataloader")
        self.sharing_secret=kwargs.get("sharing_secret")
        self.ip=kwargs.get("ip")


        # 需要提前初始化一下model


        #0代表准备状态，此时向服务端请求模型初始参数。
        #1代表本地训练状态，此时向服务端请求模型初始参数。
        #2代表发送模型参数后的再次等待训练状态(包括未被选中)
        #
        if(self.model=="CNN_MNIST"):
            self.model_structure=CNN_MNIST()
            self.cost = torch.nn.CrossEntropyLoss()


    def train(self,model_structure,optimizer):
        print(f"self.model_structure id{id(self.model_structure) }")
        print(f"model_structure id{id(model_structure)}")
        model_structure=model_structure.cpu()
        start_time=time.time()
        for iter in range(self.epoch):
            for data in self.traindataloader:

                optimizer.zero_grad()
                imgs, targets = data

                preds = model_structure(imgs)
                loss = self.cost(preds, targets)
                loss.backward()
                optimizer.step()
            print(f"{self.name}在第{self.round}个通信轮次训练第{iter}epoch")
        end_time=time.time()
        return (end_time-start_time)/self.epoch

    #def upload_accuracy(self,loss,acc):



    def run(self):
        while True:#根据当前的状态判断操作
            if(self.status==-1):#预备状态，向管理端发送任务开始请求，需要发送相应的sharing_secret
                upload_url = "http://"+self.ip+":8000/clientbegin"
                upload_data = {"name":self.name,"sharing_secret":self.sharing_secret,"task_id":self.task_id}
                upload_res = requests.post(upload_url, upload_data)
                print(upload_res)
                result=json.loads(upload_res.content.decode())
                print(result)
                if(result["status"]=="begin"):
                    self.status=0


            elif(self.status==0):
                upload_url = "http://"+self.ip+":8333/index"
                upload_data = {"num_of_data":100,"model":self.model,"task":self.task_id,"name":self.name}
                ##此处是重点！我们操作文件上传的时候，接口请求参数直接存到upload_data变量里面，
                # 在请求的时候，直接作为数据传递过去
                upload_res = requests.post(upload_url,
                                           upload_data)
                result=json.loads(upload_res.content.decode())
                self.rank=result["rank"]
                parameter={}
                for key, var in result["model_weight"].items():
                    parameter[key]=torch.tensor(np.array(var))
                self.learning_rate=float(result["learning_rate"])
                self.optimizer=result["optimizer"]
                self.round=int(result["round"])
                self.model_structure.load_state_dict(parameter, strict=True)
                if(self.optimizer=="Adam"):
                    self.optimizer = torch.optim.Adam(self.model_structure.parameters(),lr=self.learning_rate)
                self.status=1

            elif(self.status==1):

                upload_url = "http://"+self.ip+":8333/query"#查询选中结果
                upload_data = {"task":self.task_id,"rank":self.rank,"round":self.round}
                upload_res = requests.post(upload_url,
                                               upload_data)
                print(upload_res)


                result=json.loads(upload_res.content.decode())

                global_parameters={}
                if(result["status"]=="train"):
                    tmp_time=self.train(self.model_structure,self.optimizer)#获取模型每一个epoch的平均运行时间
                    for  key, var in self.model_structure.state_dict().items():
                        global_parameters[key] = var.clone().tolist()
                    upload_url = "http://"+self.ip+":8333/submit_model"  # 查询选中结果
                    upload_data = json.dumps(
                        {"parameters": global_parameters, "num_of_data": 100,"epoch_time":tmp_time})
                    upload_res = requests.post(upload_url,
                                               upload_data)
                    result = json.loads(upload_res.content.decode())
                    print(f"{self.name}训练完毕")

                elif(result["status"]=="wait"):
                    time.sleep(30)

                elif(result["status"]=="update_round"):
                    self.round=result["round"]
                    parameter=result["parameter"]
                    for key, var in parameter.items():
                        parameter[key] = torch.tensor(np.array(var))
                    self.model_structure.load_state_dict(parameter, strict=True)
                    loss, acc = evaluate(self.model_structure, self.cost, self.testdataloader)
                    print(f'{self.name}号参与者的测试损失值为{loss},准确率为{acc}')
                    upload_url = "http://"+self.ip+":8333/upload_acc"  # 查询选中结果
                    upload_data = json.dumps(
                        {"acc": acc, "data_of_single_client": 100,"loss":loss})
                    upload_res = requests.post(upload_url,
                                               upload_data)
                    try:
                        result = json.loads(upload_res.content.decode())


                        if(result["result"]=="success"):
                            print("结果上传成功")
                    except Exception  as e:
                        print(e)
                        print(upload_res)



                elif(result["status"]=="end"):
                    parameter=result["parameter"]
                    for key, var in parameter.items():
                        parameter[key] = torch.tensor(np.array(var))
                    self.model_structure.load_state_dict(parameter, strict=True)
                    loss, acc = evaluate(self.model_structure, self.cost, self.testdataloader)
                    path=f'{self.name}{self.model}.pt'
                    torch.save(self.model_structure.state_dict(), path)
                    print(f'{self.name}号参与者的测试损失值为{loss},准确率为{acc}')
                    upload_url = "http://"+self.ip+":8333/upload_acc"  # 查询选中结果
                    upload_data = json.dumps(
                        {"acc": acc, "data_of_single_client": 100,"loss":loss})
                    upload_res = requests.post(upload_url,
                                               upload_data)
                    try:
                        result = json.loads(upload_res.content.decode())
                        if(result["result"]=="success"):
                            print("结果上传成功")
                    except Exception  as e:
                        print(e)
                        print(upload_res)

                    print(f"{self.name}训练结束")
                    break


            time.sleep(10)




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Standalone training example")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    args = parser.parse_args()

    num_of_client=5
    train_data = torchvision.datasets.MNIST("./mnist",
                                            train=True,
                                            download=True,
                                            transform=transforms.ToTensor())



    test_data = torchvision.datasets.MNIST("./mnist",
                                            train=False,
                                            download=True,
                                            transform=transforms.ToTensor())
    data_indices = random_slicing(train_data, num_of_client)

    to_select = [i for i in range(num_of_client)]

    print(to_select)

    trainloader_list = [
        DataLoader(
            dataset=train_data,
            batch_size=256,
            sampler=SubsetSampler(indices=data_indices[i]),
        )
        for i in to_select
    ]


    testloader_list = [
        DataLoader(
            dataset=train_data,
            batch_size=int(len(test_data)/num_of_client),
            sampler=SubsetSampler(indices=data_indices[i]),
        )
        for i in to_select
    ]


    #print(trainloader_list[0].dataset)
    clients=[]
    for i in range(num_of_client):
        clients.append(training_server(model="CNN_MNIST",name=i,task_id=1,traindataloader=trainloader_list[i],testdataloader=testloader_list[i],epoch=1,sharing_secret="NUDT",ip=args.ip))
    for i in range(num_of_client):
        clients[i].start()
        time.sleep(1)
    for i in range(num_of_client):
        clients[i].join()












