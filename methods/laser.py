import random
import shutil
import string
from pathlib import Path
from typing import List

import numpy as np
from docker import DockerClient
from docker.errors import ImageNotFound
import docker

from methods.base import EmbeddingBase


class LaserEmbedding(EmbeddingBase):

    def __init__(self):
        self.client: DockerClient = docker.from_env()
        self.__init_laser()
        self.size = 1024

    def dim(self) -> int:
        return self.size

    def batcher(self, params, batch: List[List[str]]) -> np.ndarray:
        batch = [" ".join(sent) if sent != [] else ['.'] for sent in batch]
        return self.run_laser(batch)

    def run_laser(self, texts: List[str]) -> np.ndarray:
        resources: Path = Path("resources")
        tmp_path: Path = resources / ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        tmp_path.mkdir(exist_ok=False)
        input_path: Path = tmp_path / "input.txt"
        input_path.write_text("\n".join(texts), encoding="utf-8")
        self.__run_container(tmp_path)
        output_path: Path = tmp_path / "output.npy"
        res = np.fromfile(str(output_path.absolute()), dtype=np.float32, count=-1)
        res.resize(res.shape[0] // self.size, self.size)
        shutil.rmtree(tmp_path)
        return res

    def __run_container(self, tmp_path: Path):
        docker_cmd = "bash /src/tasks/embed/embed.sh /resources/input.txt pl /resources/output.npy"
        docker_vol = {str(tmp_path.absolute()): {"bind": "/resources", "mode": "rw"}}
        self.client.containers.run("laser:latest", docker_cmd, remove=True, volumes=docker_vol)

    def __init_laser(self):
        try: self.client.images.get("laser:latest")
        except ImageNotFound:
            url = "https://github.com/sdadas/LASER.git"
            dockerfile = "Dockerfile.cpu"
            self.client.images.build(path=url, dockerfile=dockerfile, tag="laser")
