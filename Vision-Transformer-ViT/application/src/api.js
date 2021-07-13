/* Con este módulo manejaremos los request del lado del servidor y lo conectaremos
para conectar con flask */

import axios from "axios";

const baseUrl = "http://192.168.0.103:5000/ViT"; // esta parte varia

function transform(data) {
  return axios.post(baseUrl, data);
}

export { transform };
  