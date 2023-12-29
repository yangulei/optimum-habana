FROM vault.habana.ai/gaudi-docker/1.13.0/ubuntu20.04/habanalabs/pytorch-installer-2.1.0:latest

# Installs pdsh and upgrade pip
RUN apt-get update && apt-get install -y pdsh && \
   python -m pip install --upgrade pip

# Docker ssh port setup
RUN sed -i 's/#Port 22/Port 5122/g' /etc/ssh/sshd_config && \
   sed -i 's/#   Port 22/    Port 5122/g' /etc/ssh/ssh_config && \
   sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
   service ssh restart

# Installs Optimum Habana and Habana's fork of DeepSpeed
RUN pip install git+https://hub.fgit.cf/huggingface/optimum-habana.git@v1.9-release -i https://mirror.sjtu.edu.cn/pypi/web/simple && \
   pip install git+https://hub.fgit.cf/HabanaAI/DeepSpeed.git@1.13.0 -i https://mirror.sjtu.edu.cn/pypi/web/simple && \
   pip install evaluate scikit-learn peft==0.6.2 -i https://mirror.sjtu.edu.cn/pypi/web/simple

CMD ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa && \
   chmod 600 ~/.ssh/id_rsa && \
   cat ~/.ssh/id_rsa.pub > ~/.ssh/authorized_keys && \
   /bin/bash
