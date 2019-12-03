from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six import string_types
from typing import Any, List, Text, Dict, Set
from onnx import ModelProto, ValueInfoProto

import onnx.checker

def instantiate_batch_size(model, inputs, outputs, batch_size):   # type: (ModelProto, Dict[Text, int], Dict[Text, int], int) -> ModelProto
    """
    This function instantiates inputs / outputs of model with corresponding value.

    :param model: Model with abstract batch size
    :param inputs: Dictionary of input node names and zero-based Dimension index to parametrize
    :param outputs: Dictionary of output node names and zero-based Dimension index to parametrize
    :param batch_size: integer to be set for batch_size
    :return: model with batch_size as batch size
    """
    for name, val in inputs.items():
        for input in model.graph.input:
            if input.name != name:
                continue
            if len(input.type.tensor_type.shape.dim) <= val:
                raise ValueError('dimension {} to be changed out of bounds ' .format(name))
            dim = input.type.tensor_type.shape.dim[val]
            if dim.HasField('dim_value'):
                raise ValueError('input {} already has value {} as first dimension '.format(input.name, dim.dim_value))
            else:
                dim.dim_value = batch_size

    for name, val in outputs.items():
        for output in model.graph.output:
            if output.name != name:
                continue
            if len(input.type.tensor_type.shape.dim) <= val:
                raise ValueError('dimension {} to be changed out of bounds ' .format(name))
            dim = output.type.tensor_type.shape.dim[val]
            if dim.HasField('dim_value'):
                raise ValueError('input {} already has value {} as first dimension '.format(output.name, dim.dim_value))
            else:
                dim.dim_value = batch_size
    onnx.checker.check_model(model)
    return model

def abstract_batch_size(model, inputs, outputs):   # type: (ModelProto, Dict[Text, int], Dict[Text, int]) -> ModelProto
    """
    This function abstracts concrete batch sizes of input / output node in dimension val
    into a unique dim_param. Raises Error if batch size is parametric.

    :param model: Model whos inputs / outputs will be altered
    :param inputs: Dictionary of input node names and zero-based Dimension index to parametrize
    :param outputs: Dictionary of output node names and zero-based Dimension index to parametrize
    :return: Model with abstract batch size
    """

    dim_param_set = set()

    def init_dim_param_set(dim_param_set, value_infos):  # type: (Set[Text], List[ValueInfoProto]) -> None
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField('dim_param'):
                    dim_param_set.add(dim.dim_param)  # type: ignore

    init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore

    param_name = 'batch_size'

    i = 0
    while param_name in dim_param_set:
        param_name = param_name + '_' + str(i)
        i = i + 1

    for name, val in inputs.items():
        for input in model.graph.input:
            if input.name != name:
                continue
            if len(input.type.tensor_type.shape.dim) <= val:
                raise ValueError('dimension {} to be changed out of bounds ' .format(name))
            dim = input.type.tensor_type.shape.dim[val]
            if dim.HasField('dim_param'):
                raise ValueError('input {} already has parameter {} as first dimension '.format(input.name, dim.dim_param))
            else:
                dim.dim_param = param_name

    for name, val in outputs.items():
        for output in model.graph.output:
            if output.name != name:
                continue
            if len(input.type.tensor_type.shape.dim) <= val:
                raise ValueError('dimension {} to be changed out of bounds ' .format(name))
            dim = output.type.tensor_type.shape.dim[val]
            if dim.HasField('dim_param'):
                raise ValueError('input {} already has parameter {} as first dimension '.format(output.name, dim.dim_param))
            else:
                dim.dim_param = param_name
    onnx.checker.check_model(model)
    return model


def update_inputs_outputs_dims(model, input_dims, output_dims):  # type: (ModelProto, Dict[Text, List[Any]], Dict[Text, List[Any]]) -> ModelProto
    """
        This function updates the dimension sizes of the model's inputs and outputs to the values
        provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
        will be set for that dimension.

        Example. if we have the following shape for inputs and outputs:
                shape(input_1) = ('b', 3, 'w', 'h')
                shape(input_2) = ('b', 4)
                and shape(output)  = ('b', 'd', 5)

            The parameters can be provided as:
                input_dims = {
                    "input_1": ['b', 3, 'w', 'h'],
                    "input_2": ['b', 4],
                }
                output_dims = {
                    "output": ['b', -1, 5]
                }

            Putting it together:
                model = onnx.load('model.onnx')
                updated_model = update_inputs_outputs_dims(model, input_dims, output_dims)
                onnx.save(updated_model, 'model.onnx')
    """
    dim_param_set = set()  # type: Set[Text]

    def init_dim_param_set(dim_param_set, value_infos):  # type: (Set[Text], List[ValueInfoProto]) -> None
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField('dim_param'):
                    dim_param_set.add(dim.dim_param)  # type: ignore

    init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore

    def update_dim(tensor, dim, j, name):  # type: (ValueInfoProto, Any, int, Text) -> None
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField('dim_value') and dim_proto.dim_value != dim:
                    raise ValueError('Unable to set dimension value to {} for axis {} of {}. Contradicts existing dimension value {}.'
                        .format(dim, j, name, dim_proto.dim_value))
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + '_' + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError('Unable to generate unique dim_param for axis {} of {}. Please manually provide a dim_param value.'
                        .format(j, name))
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, string_types):
            dim_proto.dim_param = dim
        else:
            raise ValueError('Only int or str is accepted as dimension value, incorrect type: {}'.format(type(dim)))

    for input in model.graph.input:
        input_name = input.name
        input_dim_arr = input_dims[input_name]
        for j, dim in enumerate(input_dim_arr):
            update_dim(input, dim, j, input_name)

    for output in model.graph.output:
        output_name = output.name
        output_dim_arr = output_dims[output_name]
        for j, dim in enumerate(output_dim_arr):
            update_dim(output, dim, j, output_name)

    onnx.checker.check_model(model)
    return model
