from imagedt._backends import tensorflow_v1

tf = tensorflow_v1()


class TFmodel_Wrapper:
    """docstring for TFmodel_Wrapper"""

    def __init__(
        self,
        pbmodel_path,
        input_nodename="input",
        output_nodename="softmax",
        gpu_id=0,
        graph_name="classify",
    ):
        super().__init__()
        self.graph_name = graph_name
        self.pbmodel_path = pbmodel_path
        self.input_node = input_nodename
        self.output_node = output_nodename.split(",")
        self.gpu_id = gpu_id
        self._load_model()
        try:
            self.reset_node_name()
            self._check_node_name()
            self._set_output_node()
        except Exception:
            self.close()
            raise

    def _load_model(self):
        # easy way! load as default graph
        print(f"load tfmodel {self.pbmodel_path}")
        # with tf.device('/gpu:'+str(self.gpu_id)):
        with tf.gfile.GFile(self.pbmodel_path, "rb") as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())

        # GPU使用率
        self.config = tf.ConfigProto()  # device_count={'GPU': self.gpu_id} only gpu
        # config.gpu_options.visible_device_list= str(self.gpu_id)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 固定比例
        self.config.gpu_options.allow_growth = True

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name=self.graph_name)
        self.sess = tf.Session(graph=self.graph, config=self.config)

    def _check_node_name(self):
        # checking input and output node name, notice, notice node name error
        self.ops = self.graph.get_operations()
        all_tensor_names = [output.name for op in self.ops for output in op.outputs]
        for node_name in [self.input_node] + self.output_node:
            if not isinstance(node_name, str):
                raise ValueError(
                    f"Node name: {node_name} should be string, like: 'input' 'data' 'softmax'."
                )
            if node_name not in all_tensor_names:
                print(
                    f"##### Print some node name : {all_tensor_names[:2]}~{all_tensor_names[-2:]}. #####"
                )
                raise ValueError(
                    f"node_name: {node_name} not in graph, please check input or output node name!"
                )

    def reset_node_name(self):
        def normalize(name):
            if not isinstance(name, str):
                raise ValueError("Node names must be strings")
            prefix = self.graph_name + "/" if self.graph_name else ""
            if not name.startswith(prefix):
                name = prefix + name
            if ":" not in name:
                name += ":0"
            return name

        self.input_node = normalize(self.input_node)
        self.output_node = [normalize(name) for name in self.output_node]

    def _set_output_node(self):
        # Get handles to input and output tensors
        self.tensor_dict = {}
        intput_tensor = self.input_node
        self.image_tensor = self.graph.get_tensor_by_name(intput_tensor)
        for output_node in self.output_node:
            self.tensor_dict[output_node] = self.graph.get_tensor_by_name(output_node)

    # from imagedt.decorator import time_cost
    # @time_cost
    def predict(self, images):
        # Run inference
        output_infos = self.sess.run(
            self.tensor_dict, feed_dict={self.image_tensor: images}
        )[self.output_node[0]]
        infos = []  # get top 1 class and confidence
        for info in output_infos:
            predict_cls = int(info.argmax())
            conf = info[predict_cls]
            infos.append({"class": predict_cls, "confidence": conf})
        return infos

    # from imagedt.decorator import time_cost
    # @time_cost
    def extract(self, image):
        output_infos = self.sess.run(
            self.tensor_dict, feed_dict={self.image_tensor: image}
        )[self.output_node[1]]
        return output_infos

    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
