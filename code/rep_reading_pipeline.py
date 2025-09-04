from typing import List, Union, Optional

import numpy as np
import torch
from transformers import Pipeline

from rep_readers import DIRECTION_FINDERS, RepReader


class RepReadingPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_hidden_states(
            self,
            outputs,
            rep_token: Union[str, int] = -1,
            hidden_layers: Union[List[int], int] = -1,
            mean_pool: str = None,
            which_hidden_states: Optional[str] = None):
        
        """
        Gives the hidden states of a layer or list of layers. 
        Does it for each token. 
        Not sure when this will be used. 
        """

        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f"""{
                which_hidden_states}_hidden_states"""]

        hidden_states_layers = {}
        for layer in hidden_layers:
            # This will give (num_samples, num_sequence, embed_size) tensor
            hidden_states = outputs['hidden_states'][layer]
            if mean_pool == "mean_pooling":
                print("Performing mean pooling...")
                hidden_states = torch.mean(hidden_states, dim=1)
            else:
                hidden_states = hidden_states[:, rep_token, :]
            hidden_states_layers[layer] = hidden_states.detach()

        return hidden_states_layers

    def _sanitize_parameters(self,
                             rep_reader: RepReader = None,
                             rep_token: Union[str, int] = -1,
                             hidden_layers: Union[List[int], int] = -1,
                             component_index: int = 0,
                             mean_pool: str = None,
                             which_hidden_states: Optional[str] = None,
                             **tokenizer_kwargs):
        
        """
        Not sure what this does either.         
        """

        preprocess_params = tokenizer_kwargs
        forward_params = {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]

        assert rep_reader is None or len(rep_reader.directions) == len(
            hidden_layers), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"
        forward_params['rep_reader'] = rep_reader
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states

        return preprocess_params, forward_params, postprocess_params

    def preprocess(
            self,
            inputs: Union[str, List[str], List[List[str]]],
            **tokenizer_kwargs):
        
        """
        This tokenizes the input strings. Input can be string, list of string, list of list of string (not sure how this one works!!!!)

        """

        if self.image_processor:
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):
        return outputs

    def _forward(self, model_inputs, rep_token, hidden_layers, rep_reader=None, component_index=0, mean_pool=None,
             which_hidden_states=None):
        """
        Args:
        - which_hidden_states (str): Specifies which part of the model (encoder, decoder, or both) to compute the hidden states from.
                                    It's applicable only for encoder-decoder models. Valid values: 'encoder', 'decoder'.
        """
        # get model hidden states and optionally transform them with a RepReader


        with torch.no_grad():
            # Check if the model has encoder and decoder layers.
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):

                decoder_start_token = [
                    self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(
                    decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input

            # Ensure that 'mean_pool' is passed only to the relevant part
            outputs = self.model(**model_inputs, output_hidden_states=True)

        hidden_states = self._get_hidden_states(outputs, rep_token, hidden_layers,
                                                 mean_pool=mean_pool, which_hidden_states = which_hidden_states)
        
        # This will return the hidden states of the input strings.
        if rep_reader is None:
            return hidden_states

        return rep_reader.transform(hidden_states, hidden_layers, component_index)

    



    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, mean_pool = None, which_hidden_states: Optional[str] = None,
                                   **tokenizer_args):
        """
        Returns:
        - A dictionary where keys are layer numbers and values are the corresponding hidden states.
        """
        # Wrapper method to get a dictionary hidden states from a list of strings
        # This calls the forward function. 
        # Vectors are stored in shape (num_samples, num_layers, 1, hidden_dim) 1 corresponds to the rep_token taken out of all the outputs. 
        hidden_states_outputs = self(train_inputs, rep_token=rep_token,
                                     hidden_layers=hidden_layers, batch_size=batch_size, rep_reader=None, mean_pool = mean_pool, 
                                     which_hidden_states=which_hidden_states, **tokenizer_args)
        

        hidden_states = {layer: [] for layer in hidden_layers}
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])


        # Stacks all of them in shape (num_layers, num_samples, hidden_dim)
        return {k: np.vstack(v) for k, v in hidden_states.items()}

    def _validate_params(self, n_difference, direction_method):
        # validate params for get_directions
        if direction_method == 'clustermean':
            assert n_difference == 1, "n_difference must be 1 for clustermean"



    def get_directions(
            self,
            train_inputs: Union[str, List[str], List[List[str]]],
            rep_token: Union[str, int] = -1,
            hidden_layers: Union[str, int] = -1,
            n_difference: int = 1,
            batch_size: int = 8,
            direction_method: str = 'pca',
            direction_finder_kwargs: dict = {},
            mean_pool : str = None,
            which_hidden_states: Optional[str] = None,
            **tokenizer_args, ):
        """Train a RepReader on the training data.
        Args:
            train_inputs: list of input strings for training
            rep_token: still -1 for some reason
            n_difference: number of contrastive difference. In our case it's one (betweeen negative and positive) 
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """


        # convert hidden layers to a list
        if not isinstance(hidden_layers, list):
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]


        # Something specific for clustermean direction method. Not relevant for PCA
        self._validate_params(n_difference,
                              direction_method)

        # initialize a DirectionFinder
        direction_finder = DIRECTION_FINDERS[direction_method](
            **direction_finder_kwargs)

        # if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            # hidden_states of shape (num_layers ,num_samples, hidden_dim)
            hidden_states = self._batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, batch_size, mean_pool,
                                                            which_hidden_states, **tokenizer_args)
            

            # get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in
                                      hidden_states.items()}
            


        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers)
        
        

        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(
                    np.float32)


        return direction_finder
