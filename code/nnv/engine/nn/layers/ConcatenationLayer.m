classdef ConcatenationLayer < handle
    % Concatenation Layer object
    % Concatenate arrays along a specific dimension
    % Typically use to concatenate 1D or 2D features  (vector/matrices, rather than images such as in DepthConcatenation)
    % see: https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.concatenationlayer.html
    % 
    % Author: Diego Manzanas Lopez
    % Date: 03/15/2023
    
    properties
        Name = 'ConcatLayer';
        NumInputs = 1; % default
        InputNames = {'in'}; % default
        NumOutputs = 1; % default
        OutputNames = {'out'}; % default
        Dim = 0 % unspecified, must be a positive integer
    end
    
    methods % constructor
        
        % create layer
        function obj = ConcatenationLayer(varargin)
            % @name: name of the layer
            % @NumInputs: number of inputs
            % @NumOutputs: number of outputs,
            % @InputNames: input names
            % @OutputNames: output names
            % @Dim: dimension for concatenating inputs
            
            switch nargin
                
                case 6
                    name = varargin{1};
                    numInputs = varargin{2};
                    numOutputs = varargin{3};
                    inputNames = varargin{4};
                    outputNames = varargin{5};
                    dim = varargin{6};
                case 2
                    name = varargin{1};
                    dim = varargin{2};
                    numInputs = 1;
                    numOutputs = 1;
                    inputNames = {'in'};
                    outputNames = {'out'};
                otherwise
                    error('Invalid number of input arguments, should be 2 or 6');        
            end
            
            if ~ischar(name)
                error('Invalid name, should be a charracter array');
            end
            
            if numInputs < 1
                error('Invalid number of inputs');
            end
                       
            if numOutputs < 1
                error('Invalid number of outputs');
            end
            
            if ~iscell(inputNames)
                error('Invalid input names, should be a cell');
            end
            
            if ~iscell(outputNames)
                error('Invalid output names, should be a cell');
            end

            if ~isscalar(dim) || dim < 1
                error('Concat dimension must be a positive integer')
            end
            
            obj.Name = name;
            obj.NumInputs = numInputs;
            obj.NumOutputs = numOutputs;
            obj.InputNames = inputNames;
            obj.OutputNames = outputNames; 
            obj.Dim = dim;
        end
            
    end
        
        
    methods % main methods
        
        % evaluate
        function outputs = evaluate(obj, inputs)
            % concatenation layers takes usually two inputs, but allows many (N)
            %
            % first input is obj, the second is a cell array containing the inputs to concatenate
            % Initialize outputs as the first one
            outputs = inputs{1};
            % Concatenate the inputs 
            for k=2:length(inputs)
                outputs = cat(obj.Dim, outputs, inputs{k});
            end
                
        end
 
        % reach
        function outputs = reach_single_input(obj, inputs)
            % @in_image: input imagestar
            % @image: output set
            % outputs = inputs{1};
            % concatenate along the dimension (obj.Dim), usually channels (3)
            
            % Get max size of V
            vSize = size(inputs{1}.V);
            indexMax = 1;
            for i = 2:length(inputs)
                if numel(inputs{i}.V) > prod(vSize)
                    vSize = size(inputs{i}.V);
                    indexMax = i;
                end
            end
            
            % Initialize V
            new_V = zeros(vSize, 'like', inputs{1}.V);
            new_V(:,:,:,1:size(inputs{1}.V,4)) = inputs{1}.V;
            % Generate V
            for i = 2:length(inputs)
                % ensure all Vs are same dimension
                tempV = zeros(vSize, 'like', inputs{1}.V);
                tempV(:,:,:,1:size(inputs{i}.V,4)) = inputs{i}.V;
                new_V = cat(obj.Dim, new_V, tempV);
            end

            % Create output set
            outputs = ImageStar(new_V, inputs{indexMax}.C, inputs{indexMax}.d, inputs{indexMax}.pred_lb, inputs{indexMax}.pred_ub);
        end
        
        % reachability analysis with multiple inputs
        function IS = reach(varargin)
            % @in_image: an input imagestar
            % @image: output set
            % @option: = 'single' or 'parallel' 
           
            switch nargin
                
                 case 7
                    obj = varargin{1};
                    in_images = varargin{2};
                    method = varargin{3};
                    option = varargin{4};
                
                case 6
                    obj = varargin{1};
                    in_images = varargin{2};
                    method = varargin{3};
                    option = varargin{4};
                
                case 5
                    obj = varargin{1};
                    in_images = varargin{2};
                    method = varargin{3};
                    option = varargin{4};
                
                case 4
                    obj = varargin{1};
                    in_images = varargin{2}; 
                    method = varargin{3};
                    option = varargin{4}; % computation option

                case 3
                    obj = varargin{1};
                    in_images = varargin{2}; % don't care the rest inputs
                    method = varargin{3};
                    option = [];

                otherwise
                    error('Invalid number of input arguments (should be 2, 3, 4, 5 or 6)');
            end
            
            if strcmp(method, 'approx-star') || strcmp(method, 'exact-star') || strcmp(method, 'abs-dom') || strcmp(method, 'approx-zono') || contains(method, "relax-star")
                IS = obj.reach_single_input(in_images);
            else
                error('Unknown reachability method');
            end
  
        end
        
    end

    
    methods % helper method

        % change params to gpuArrays
        function obj = toGPU(obj)
            % nothing to change in here (no params)
        end

        % Change params precision
        function obj = changeParamsPrecision(obj, ~)
            % nothing to change in here (no params)
        end
        
    end
    
    
    methods(Static)
        
        % parsing method
        function L = parse(layer)
            % create NNV layer from matlab
                      
            if ~isa(layer, 'nnet.cnn.layer.ConcatenationLayer')
                error('Input is not a concatenation layer');
            end
            L = ConcatenationLayer(layer.Name, layer.NumInputs, layer.NumOutputs, layer.InputNames, layer.OutputNames, layer.Dim);
        end

    end
end

