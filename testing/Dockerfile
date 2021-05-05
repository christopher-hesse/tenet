FROM christopherhesse/dockertest:v5

RUN curl -o zig.tar.xz https://ziglang.org/download/0.7.1/zig-linux-x86_64-0.7.1.tar.xz
RUN mkdir /opt/zig
RUN tar xvf zig.tar.xz -C /opt/zig --strip-components=1
ENV PATH="${PATH}:/opt/zig"

# ADD env.yaml .
# RUN conda env update --name env --file env.yaml