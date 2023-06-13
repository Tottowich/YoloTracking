from encodings import normalize_encoding
import socket
import json
import torch
import numpy as np
from mlsocket import MLSocket
import time
import threading
class Transmitter():
    """
    Trasmitter using  mlsocket.
    """
    def __init__(self, 
                    reciever_ip:str, 
                    reciever_port:int,
                    ml_reciever_ip:str=None,
                    ml_reciever_port:int=None,
                    classes_to_send=None
                    ):
        """
        Args:
            reciever_ip: IP of the reciever.
            reciever_port: Port of the reciever.
        """
        super().__init__()
        self.reciever_ip = reciever_ip
        self.reciever_port = reciever_port
        self.ml_reciever_ip = ml_reciever_ip
        self.ml_reciever_port = ml_reciever_port

        self.classes_to_send = classes_to_send
        self.pcd = None
        self.s_ml = MLSocket()
        self.s_udp = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.udp_connect = False
        self.pred_dict = None
        self.started_udp = False
        self.started_ml = False
        self.thread_udp = None
        self.thread_udp = None
        self.previous_det = 0

        #try:
        #    self.sock.connect(self.reciever_ip)
        #except:
        #    print("Could not connect to the reciever.")

    def send_dict(self):
        """
        Send a dictionary.
        Args:
            dict: Dictionary to send.
        
        """
        # if len(self.pred_dict["pred_labels"]) == 0:
        #     try:
        #         self.s_udp.sendto(self.pred_dict, (self.reciever_ip, self.reciever_port))
        #         print()
        #     except:
        #         pass
        #     return
        if len(self.pred_dict["pred_labels"]) > 0:
            if isinstance(self.pred_dict["pred_labels"],torch.Tensor):
                self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"].cpu().numpy()
            if isinstance(self.pred_dict["pred_boxes"],torch.Tensor):
                self.pred_dict["pred_boxes"] = self.pred_dict["pred_boxes"].cpu().numpy()
            if isinstance(self.pred_dict["pred_scores"],torch.Tensor):
                self.pred_dict["pred_scores"] = self.pred_dict["pred_scores"].cpu().numpy()
            self.pred_dict["pred_boxes"] = self.pred_dict["pred_boxes"].reshape(self.pred_dict["pred_boxes"].shape[0],-1).tolist()
            self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"].reshape(self.pred_dict["pred_labels"].shape[0],-1).tolist()
            self.pred_dict["pred_scores"] = self.pred_dict["pred_scores"].reshape(self.pred_dict["pred_scores"].shape[0],-1).tolist()
        else:
            self.pred_dict["pred_boxes"] = self.pred_dict["pred_boxes"].tolist()
            self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"].tolist()
            self.pred_dict["pred_scores"] = self.pred_dict["pred_scores"].tolist()
        #if self.classes_to_send is not None:
        #
        #    indices = [np.nonzero(sum(self.pred_dict["pred_labels"]==x for x in self.classes_to_send))[0].tolist()][0]
        #    print(f"self.pred_dict['pred_labels'].shape: {self.pred_dict['pred_labels'].shape}")
        #    self.pred_dict["pred_boxes"] = self.pred_dict["pred_boxes"][indices,:].tolist()
        #    self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"][indices,:].tolist()
        #    self.pred_dict["pred_scores"] = self.pred_dict["pred_scores"][indices,:].tolist()
        """
            for i,v in enumerate(self.pred_dict["pred_labels"]):
                print(v)
                if v in self.classes_to_send:
                    dict_selective['pred_boxes'].append(self.pred_dict["pred_boxes"][i])
                    dict_selective['pred_labels'].append(self.pred_dict["pred_labels"][i])
                    dict_selective['pred_scores'].append(self.pred_dict["pred_scores"][i])
            self.pred_dict = dict_selective
        dict["pcd"] = pcd.tolist()
        """
        predictions_encoded = json.dumps(self.pred_dict).encode('utf-8')
        try:
            self.s_udp.sendto(predictions_encoded, (self.reciever_ip, self.reciever_port))
        except:
            print(f"Could not send to {self.reciever_ip}")
        self.pred_dict = None
    def send_pcd(self):
        if isinstance(self.pred_dict["pred_labels"],torch.Tensor):
            self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"].cpu().numpy()
        #indices = np.nonzero(sum(self.pred_dict["pred_labels"]==x for x in self.classes_to_send))[0].tolist()
        pred_send = np.concatenate((self.pred_dict["pred_boxes"],self.pred_dict["pred_labels"],self.pred_dict["pred_scores"]),axis=1)
        
        #pred_send = np.concatenate((self.pred_dict["pred_boxes"][indices,:],self.pred_dict["pred_labels"][indices,:],self.pred_dict["pred_scores"][indices,:]),axis=1)
        self.s_ml.send(self.pcd)
        self.s_ml.send(pred_send)
        self.pred_dict = None
        self.pcd = None
    def start_transmit_ml(self):
        """
        Start the transmission.
        """
        try: 
            self.s_ml.connect((self.ml_reciever_ip,self.ml_reciever_port))
            self.started_ml = True
            """
            self.thread_udp = threading.Thread(target=self.transmit_udp)
            self.thread_udp.daemon = True
            self.thread_udp.start()
            """
            
            return True
        except:
            return False
    def stop_transmit_ml(self):
        if self.started_ml:
            self.started_ml = False
            #if self.thread_ml is not None:
            #    self.thread_ml.join()
        
    def _check_connection(self):
        """
        Check if the connection is alive.
        """
        try:
            self.sock.sendto(b'ping', (self.reciever_ip, self.reciever_port))
            data, addr = self.sock.recvfrom(1024)
            if data.decode('utf-8') == 'pong':
                return True
            else:
                return False
        except:
            return False
    def check_connection(self):
        """
        Check if the connection is alive.
        """
        try:
            self.connect((self.reciever_ip, self.reciever_port))
        except:
            return False
    def start_transmit_udp(self):
        """
        Start the transmission.
        """
        try: 
            self.s_udp.connect((self.reciever_ip,self.reciever_port))
            self.started_udp = True
            """
            self.thread_udp = threading.Thread(target=self.transmit_udp)
            self.thread_udp.daemon = True
            self.thread_udp.start()
            
            """
            return True
        except:
            return False
    def stop_transmit_udp(self):
        """
        Stop the transmission.
        """
        if self.started_udp:
            self.started_udp = False
            #if self.thread_udp is not None:
            #    self.thread_udp.join()
    def transmit_udp(self):
        while self.started_udp:
            if self.pred_dict is not None:
                self.send_dict()
        
    def transmit_ml_socket(self):
        """
        Transmit the data via MLSocket.
        """
        print("Started Transmitter to {}:{}".format(self.reciever_ip, self.reciever_port))
        while self.started_ml:
            if self.pcd is not None and self.pred_dict is not None:
                print("SENDING DATA")
                self.s.send(self.pcd)
                try:
                    for key, value in (self.pred_dict.items()):
                        if isinstance(value,torch.Tensor):
                            self.pred_dict[key] = value.detach().cpu().numpy()
                    if self.classes_to_send is not None:
                        indices = self.pred_dict["pred_labels"] in self.classes_to_send
                    pred_arr = np.concatenate([np.array(self.pred_dict["pred_boxes"][:,:6]),np.array(self.pred_dict["pred_labels"]),np.array(self.pred_dict["pred_scores"])],axis=1)
                    self.s.send(pred_arr)
                    print(f"SENT : {pred_arr.shape}")
                except:
                    pass
                self.pcd = None
                self.pred_dict = None
def test_multiport_transmitter(host_udp:str=None,
                               port_udp:int=None,
                               host_ml:str=None,
                               port_ml:int=None):
    """
    Test the multiport transmitter.
    Args:
        host_udp: The host (ip-adress) of the udp socket.
        port_udp: The port of the udp socket.
        host_ml: The host (ip-adress) of the ml socket.
        port_ml: The port of the ml socket.
    """
    transmitter = Transmitter(host_udp,port_udp,host_ml,port_ml)
    transmitter.start_transmit_udp()
    transmitter.start_transmit_ml()
    # Generate random point cloud, bounding box, labels and scores:
    detections = 10
    pcd = np.random.random((100000,3))
    bboxes = np.random.random((detections,3))
    lbls = np.random.randint(0,10,(detections,1))
    score = np.random.randint(0,10,(detections,1))
    # dictionary of bounding boxes, labels and scores:
    pred_dict = {"pred_boxes":bboxes,"pred_labels":lbls,"pred_scores":score}
    transmitter.pred_dict = pred_dict
    # Send the data:
    transmitter.pcd = pcd
    transmitter.send_pcd()
    transmitter.send_dict()

    # Wait for the data to be received:
    transmitter.stop_transmit_udp()
    transmitter.stop_transmit_ml()
    transmitter.close()
    
if __name__ == "__main__":
    test_multiport_transmitter("192.168.200.103",7002,"192.168.200.103",1234)



