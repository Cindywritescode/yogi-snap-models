#!/bin/bash

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='cindys-poses' \
    --saved_model_tags=serve \
    ./saved_model/experiment/improved_model \
    ./saved_model/web_model