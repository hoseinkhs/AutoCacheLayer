from locust import task, between
import locust_plugins
from locust_plugins.users import SocketIOUser
import json
import time
from locust.exception import StopUser
class LoadGenUser(SocketIOUser):
    user_num =0
    def __init__(self, *args, **kwargs):
        super(LoadGenUser, self).__init__(*args, **kwargs)
        self.connect("ws://localhost:9090/ws")
        self.done = False
        self.TL = 1000
        self.num = LoadGenUser.user_num
        LoadGenUser.user_num+=1

    @task
    def predict(self):
        if self.done:
            return
        self.done = True
        
        self.start_time = time.perf_counter()
        print("User:", self.num)
        self.send(json.dumps({"id": str(self.num), "count": 100}), name="predict")
        
    def on_message(self, msg):
        msg = json.loads(msg)
        # print("msg received:", msg, "by", self.num)
        self.response_time = (time.perf_counter()-self.start_time) *1000 #* msg["T"]
        # print(self.response_time)
        if self.response_time <= self.TL:
            self.environment.events.request.fire(
                request_type="WSR",
                name="predict",
                response_time= self.response_time,
                response_length=len(msg),
                exception=None,
                context=self.context(),
            )
        else:
            self.environment.events.request.fire(
                request_type="WSR",
                name="predict",
                response_time= self.response_time,
                response_length=len(msg),
                exception=Exception("TLE"),
                context=self.context(),
            )
        # raise StopUser()
