"""
Callback server
"""
import uvicorn
import torch
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from torchvision import transforms
from PIL import Image


class CallbackServer:
    @staticmethod
    def get_tensor_image(img_buff):
        transform = transforms.Compose([transforms.ToTensor()])
        img = Image.frombytes(mode="RGB", size=(256, 256), data=img_buff)
        img = transform(img)
        img = img[0, :, :].unsqueeze(0)
        return img.unsqueeze(0)

    @staticmethod
    def start(callback):
        """
        Function of start http server
        """
        fapi = FastAPI()

        @fapi.post(
            "/",
            responses={
                200: {
                    "content": {"application/json": {"example": {}}}
                }
            }
        )
        def execute_oneshot(
            image_feats: List[UploadFile] = File(description="four magnetogram images"),
            physical_feats: List[str] = Form(description="physical features")
        ):
            imgs = torch.cat([CallbackServer.get_tensor_image(io.file.read()) for io in image_feats])
            phys = np.array([list(map(float, raw.split(","))) for raw in physical_feats])[:, :90]
            prob = callback(imgs, phys).tolist()
            return JSONResponse(content={"probability": {"OCMX"[i]: prob[i] for i in range(len(prob))}})

        host_name = "127.0.0.1"
        port_num = 8080
        uvicorn.run(fapi, host=host_name, port=port_num)
