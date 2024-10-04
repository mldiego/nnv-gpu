classdef FullyConnectedLayer < handle
    % The FullyConnectedLayer layer class in CNN
    %   Contain constructor and reachability analysis methods
    % Main references:
    % 1) An intuitive explanation of convolutional neural networks: 
    %    https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
    % 2) More detail about mathematical background of CNN
    %    http://cs231n.github.io/convolutional-networks/
    %    http://cs231n.github.io/convolutional-networks/#pool
    % 3) Matlab implementation of Convolution2DLayer and MaxPooling (for training and evaluating purpose)
    %    https://www.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html
    %    https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html
    
    %   Dung Tran: 6/26/2019
    
    properties
        Name = 'fully_connected_layer';
        % Hyperparameters
        InputSize = 0;  % number of input
        OutputSize = 0; % number of output
        Weights = []; % weight matrix
        Bias  = []; % bias vector     
    end
    
    
    methods % main methods
        
        % constructor of the class
        function obj = FullyConnectedLayer(varargin)
            % author: Dung Tran
            % date: 6/26/2019    
            % update: 
            
            switch nargin
                
                case 3
                    name = varargin{1};
                    W = varargin{2};
                    b = varargin{3};
                    if ~ischar(name)
                        error('Name is not char');
                    else
                        obj.Name = name;
                    end
                    if size(W, 1) ~= size(b, 1)
                        error('Inconsistent dimension between the weight matrix and bias vector');
                    end
                    if size(b,2) ~= 1
                        error('Bias vector should have one column');
                    end
                    
                    obj.InputSize = size(W,2);
                    obj.OutputSize = size(W,1);
                    obj.Weights = W;
                    obj.Bias = b;
                    
                case 2
                    
                    W = varargin{1};
                    b = varargin{2};
                    obj.Name = 'fully_connected_layer';
                    if size(W, 1) ~= size(b, 1)
                        error('Inconsistent dimension between the weight matrix and bias vector');
                    end
                    if size(b,2) ~= 1
                        error('Bias vector should have one column');
                    end
                    obj.InputSize = size(W,2);
                    obj.OutputSize = size(W,1);
                    obj.Weights = W;
                    obj.Bias = b;
                           
                case 0
                    
                    obj.Name = 'fully_connected_layer';
                    % Hyperparameters
                    obj.InputSize = 0;
                    obj.OutputSize = 0; % step size for traversing input
                    obj.Weights = [];
                    obj.Bias = [];
                    
                otherwise
                    error('Invalid number of inputs (should be 0, 2 or 3)');
            end 
             
        end
        
        % evaluation method
        function y = evaluate(obj, x)
            % @input: input
            % @y: output
            
            % author: Dung Tran
            % date: 6/26/2019
            % 
            % update: add support for sequence evaluation (neuralode, RNN)
            %   -date: 03/17/2023 (Diego Manzanas)
            
            n = size(x);
            if length(n) == 2
                x = reshape(x, [n(1)*n(2) 1]);
            elseif length(n) == 3
                x = reshape(x, [n(1)*n(2)*n(3) 1]);
            elseif length(n) == 4
                x = reshape(x, [n(1)*n(2)*n(3)*n(4) 1]);
            else
                error('Invalid input');
            end

            if size(x, 1) ~= size(obj.Weights, 2)
                error('Inconsistent input vector')
            end

            y1 = obj.Weights * x;
            if size(x, 2) ~= 1
                n = size(x, 2);
                for i=1:n
                    y1(:,i) = y1(:,i) + obj.Bias;
                end
                y = y1;
            else
                y = y1 + obj.Bias;
            end   
             
        end

        % main reachability analysis function
        function IS = reach(varargin)
            % @in_image: an input imagestar
            % @image: output set
            % @option: = 'single' or 'parallel' 
            
            % author: Dung Tran
            % date: 6/26/2019
            % update: 1/6/2020  update reason: add zonotope method
             
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
                    error('Invalid number of input arguments (should be 2, 3, 4, 5, or 6)');
            end
            
            if strcmp(method, 'approx-star') || strcmp(method, 'exact-star') || strcmp(method, 'abs-dom') || contains(method, "relax-star")
                IS = obj.reach_star_multipleInputs(in_images, option);
            else
                error('Unknown reachability method');
            end
            
        end

        % change params to gpuArrays
        function obj = toGPU(obj)
            obj.Weights = gpuArray(obj.Weights);
            obj.Bias = gpuArray(obj.Bias);
        end

        % Change params precision
        function obj = changeParamsPrecision(obj, precision)
            if strcmp(precision, "double")
                obj.Weights = double(obj.Weights);
                obj.Bias = double(obj.Bias);
            elseif strcmp(precision, "single")
                obj.Weights = single(obj.Weights);
                obj.Bias = single(obj.Bias);
            else
                error("Parameter numerical precision must be 'single' or 'double'");
            end
        end
    
    end   
     
    methods % reachability methods
        
        % reachability analysis using imagestar
        function image = reach_star_single_input(obj, in_image)
            % @in_image: input imagestar
            % @image: output set
            
            if ~isa(in_image, 'ImageStar') && ~isa(in_image, 'Star')
                error('Input set is not an ImageStar or Star');
            end
            
            if isa(in_image, 'ImageStar')
                % reach using ImageStar
                N = in_image.height*in_image.width*in_image.numChannel;
                if N~= obj.InputSize
                    error('Inconsistency between the size of the input image and the InputSize of the network');
                end
                           
                n = in_image.numPred;
                V(1, 1, :, in_image.numPred + 1) = zeros(obj.OutputSize, 1, 'like', in_image.V);
                V(1, 1, :, :) = obj.Weights*reshape(in_image.V, N, n + 1);
                V(1, 1, :, 1) = reshape(V(1, 1, :, 1), obj.OutputSize, 1) + obj.Bias;
                
                % output set
                image = ImageStar(V, in_image.C, in_image.d, in_image.pred_lb, in_image.pred_ub);

            else % reach Star set
                image = in_image.affineMap(obj.Weights, obj.Bias);

            end
            
        end
        
        % handle multiple inputs
        function S = reach_star_multipleInputs(obj, inputs, option)
            % @inputs: an array of ImageStars
            % @option: = 'parallel' or 'single'
            % @S: output ImageStar
            
            n = length(inputs);
            if isa(inputs, "ImageStar")
                S(n) = ImageStar;
            elseif isa(inputs, "Star")
                S(n) = Star;
            else
                error("Input must be ImageStar or Star");
            end
            
            if strcmp(option, 'parallel')
                parfor i=1:n
                    S(i) = obj.reach_star_single_input(inputs(i));
                end
            elseif strcmp(option, 'single') || isempty(option)
                for i=1:n
                    S(i) = obj.reach_star_single_input(inputs(i));
                end
            else
                error('Unknown computation option, should be parallel or single');
            end
            
        end
        
    end
    
    
    methods(Static)
        
        % parse a trained FullyConnectedLayer from matlab
        function L = parse(fully_connected_layer)
            % @fully_connecteted_Layer: a fully connected layer from matlab deep
            % neural network tool box
            % @L : a FullyConnectedLayer obj for reachability analysis purpose
            
            if ~isa(fully_connected_layer, 'nnet.cnn.layer.FullyConnectedLayer')
                error('Input is not a Matlab nnet.cnn.layer.FullyConnectedLayer class');
            end
            L = FullyConnectedLayer(fully_connected_layer.Name, fully_connected_layer.Weights, fully_connected_layer.Bias);         
            
        end
        
    end
    
end

