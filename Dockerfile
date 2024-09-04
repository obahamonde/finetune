# Start from the AWS Glue base image
FROM amazon/aws-glue-libs:glue_libs_3.0.0_image_01

USER root

# Update system and install dependencies
RUN yum update -y && \
    yum install -y gcc bzip2-devel libffi-devel zlib-devel wget make iptables kernel-devel-$(uname -r) kernel-headers-$(uname -r)

# Install Python 3.10 and dependencies
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make altinstall

# Set Python 3.10 as the default version
RUN alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

# Install pip for Python 3.10 and upgrade it
RUN /usr/local/bin/python3.10 -m ensurepip && \
    /usr/local/bin/python3.10 -m pip install --upgrade pip

# Insta# Instalar CUDA Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo -O /etc/yum.repos.d/cuda-rhel7.repo && \
    yum clean all && \
    yum -y install cuda-toolkit-12-2

# Configurar variables de entorno para CUDA
ENV PATH=/usr/local/cuda-12.2/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}

# Switch to glue_user
USER glue_user

WORKDIR /app

# Copy your application code
COPY . .

# Install application dependencies from requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Expose necessary ports
EXPOSE 8888 8000

# Set the entrypoint
ENTRYPOINT ["make", "dev"]