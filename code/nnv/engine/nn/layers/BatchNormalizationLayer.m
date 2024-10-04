classdef BatchNormalizationLayer < handle
    % The Batch Normalization Layer class 
    %   Contain constructor and reachability analysis methods   
    %   Dung Tran: 1/1/2020
    
    properties
        Name = 'BatchNormalizationLayer';
        
        NumChannels = [];
        TrainedMean = [];
        TrainedVariance = [];
        
        % Hyperparameters 
        Epsilon = 0.00001;  % default value
        
        % Learnable parameters
        Offset = [];
        Scale = [];
        
        % layer properties
        NumInputs = 1;
        InputNames = {'in'};
        NumOutputs = 1;
        OutputNames = {'out'};
               
    end
    
    
    % setting hyperparameters method
    methods
        
        % constructor of the class
        function obj = BatchNormalizationLayer(varargin)
            % author: Dung Tran
            % date: 1/1/2020    
            
            if mod(nargin, 2) ~= 0
                error('Invalid number of arguments');
            end
            
            for i=1:nargin-1
                
                if mod(i, 2) ~= 0
                    
                    if strcmp(varargin{i}, 'Name')
                        obj.Name = varargin{i+1};
                    elseif strcmp(varargin{i}, 'NumChannels')
                        obj.NumChannels = varargin{i+1};
                    elseif strcmp(varargin{i}, 'TrainedMean')
                        obj.TrainedMean = double(varargin{i+1});
                    elseif strcmp(varargin{i}, 'TrainedVariance')
                        obj.TrainedVariance = double(varargin{i+1});
                    elseif strcmp(varargin{i}, 'Epsilon')
                        obj.Epsilon = double(varargin{i+1});
                    elseif strcmp(varargin{i}, 'Offset')
                        obj.Offset = double(varargin{i+1});
                    elseif strcmp(varargin{i}, 'Scale')
                        obj.Scale = double(varargin{i+1});
                    elseif strcmp(varargin{i}, 'NumInputs')
                        obj.NumInputs = double(varargin{i+1});
                    elseif strcmp(varargin{i}, 'InputNames')
                        obj.InputNames = varargin{i+1};
                    elseif strcmp(varargin{i}, 'NumOutputs')
                        obj.NumOutputs = double(varargin{i+1});
                    elseif strcmp(varargin{i}, 'OutputNames')
                        obj.OutputNames = varargin{i+1};
                    end
                    
                end
                if isempty(obj.Offset)
                    obj.Offset = zeros(1,obj.NumChannels);
                end
                if isempty(obj.Scale)
                    obj.Scale = ones(1,obj.NumChannels);
                end
                
            end
                
             
        end
        
        % change params to gpuArrays
        function obj = toGPU(obj)
            obj.Offset = gpuArray(obj.Offset);
            obj.Scale = gpuArray(obj.Scale);
            obj.Epsilon = gpuArray(obj.Epsilon);
            obj.TrainedMean = gpuArray(obj.TrainedMean);
            obj.TrainedVariance = gpuArray(obj.TrainedVariance);
        end

        % Change params precision
        function obj = changeParamsPrecision(obj, precision)
            if strcmp(precision, "double")
                obj.Offset = double(obj.Offset);
                obj.Scale = double(obj.Scale);
                obj.Epsilon = double(obj.Epsilon);
                obj.TrainedMean = double(obj.TrainedMean);
                obj.TrainedVariance = double(obj.TrainedVariance);
            elseif strcmp(precision, "single")
                obj.Offset = single(obj.Offset);
                obj.Scale = single(obj.Scale);
                obj.Epsilon = single(obj.Epsilon);
                obj.TrainedMean = single(obj.TrainedMean);
                obj.TrainedVariance = single(obj.TrainedVariance);
            else
                error("Parameter numerical precision must be 'single' or 'double'");
            end
        end

    end
        
    
    methods % evaluation functions
        
        % Evaluate input 
        function y = evaluate(obj, input)
            % @input: input image
            % @y: output image with normalization
            
            % author: Mykhailo Ivashchenko
            % date: 9/17/2022
            
            if(length(size(input)) == 2) && size(obj.TrainedMean, 1) == 1 && size(obj.TrainedMean, 2) == 1 && length(size(obj.TrainedMean)) == 3
                obj.TrainedMean = reshape(obj.TrainedMean, [size(obj.TrainedMean, 3) 1]);
                obj.TrainedVariance = reshape(obj.TrainedVariance, [size(obj.TrainedVariance, 3) 1]);
                obj.Epsilon = reshape(obj.Epsilon, [size(obj.Epsilon, 3) 1]);
                obj.Offset = reshape(obj.Offset, [size(obj.Offset, 3) 1]);
                obj.Scale = reshape(obj.Scale, [size(obj.Scale, 3) 1]);
                obj.NumChannels = 1;
            end
            
            % input -> 3d image / volume
            if length(size(input)) == 4 || (length(size(input)) == 3 && obj.NumChannels == 1)
                if ~isempty(obj.TrainedMean) && ~isempty(obj.TrainedVariance) && ~isempty(obj.Epsilon) && ~isempty(obj.Offset) && ~isempty(obj.Scale)
                    y = input - obj.TrainedMean;
                    for i=1:obj.NumChannels
                        y(:,:,:,i) = y(:,:,:,i)./(sqrt(obj.TrainedVariance(:,:,:,i) + obj.Epsilon));
                        y(:,:,:,i) = obj.Scale(:,:,:,i).*y(:,:,:,i) + obj.Offset(:,:,:,i);
                    end
                elseif ~isempty(obj.Scale) && ~isempty(obj.Offset) && isempty(obj.TrainedVariance) && isempty(obj.TrainedMean)
                    y = input;
                    if obj.NumChannels == 1
                        y = obj.Scale.*y + obj.Offset;
                    else
                        for i=1:obj.NumChannels
                            y(:,:,:,i) = obj.Scale(:,:,:,i).*y(:,:,:,i) + obj.Offset(:,:,:,i);
                        end
                    end
                else
                    y = input;
                end
            % input => image or feature vector
            else
                if ~isempty(obj.TrainedMean) && ~isempty(obj.TrainedVariance) && ~isempty(obj.Epsilon) && ~isempty(obj.Offset) && ~isempty(obj.Scale)
                    y = input - obj.TrainedMean;
                    for i=1:obj.NumChannels
                        y(:,:,i) = y(:,:,i)./(sqrt(obj.TrainedVariance(:,:,i) + obj.Epsilon));
                        y(:,:,i) = obj.Scale(:, :, i).*y(:,:,i) + obj.Offset(:,:,i);
                    end
                elseif ~isempty(obj.Scale) && ~isempty(obj.Offset) && isempty(obj.TrainedVariance) && isempty(obj.TrainedMean)
                    y = input;
                    if obj.NumChannels == 1
                        y = obj.Scale.*y + obj.Offset;
                    else
                        for i=1:obj.NumChannels
                            y(:,:,i) = obj.Scale(:, :, i).*y(:,:,i) + obj.Offset(:,:,i);
                        end
                    end
                else
                    y = input;
                end
            end
            
        end 
    
    end
    
    
    methods % reachability functions
                
        % reachability method of a single star (in_image is a 1x1 ImageStar)
        function image = reach_star_single_input(obj, in_image)
            % @in_image: an input imagestar
            % @image: output set
            
            if ~isa(in_image, 'ImageStar') && ~isa(in_image, 'Star') && ~isa(in_image, 'VolumeStar') % input may be Star or ImageStar
                error('Input is not a Star, ImageStar or VolumeStar');
            end
            
            % make copy of input set
            image = in_image;

            if isa(image, 'VolumeStar')
                
                % Begin reachability analysis
                x = in_image;
                
                % If mean and variance exist
                if ~isempty(obj.TrainedMean) && ~isempty(obj.TrainedVariance) && ~isempty(obj.Epsilon) && ~isempty(obj.Offset) && ~isempty(obj.Scale)
                    % 1) Normalized all elements of x (for each channel independently)
                    % 1a) substract mean from elements of x per channel
                    for i=1:obj.NumChannels
                        x.V(:,:,:,i,1) = x.V(:,:,:,i,1) - obj.TrainedMean(:,:,:,i);
                    end
                    % 1b) Divide by sqrt(variance + epsilon)
                    for i=1:obj.NumChannels
                        x.V(:,:,:,i,:) = x.V(:,:,:,i,:) .* 1/sqrt(obj.TrainedVariance(:,:,:,i) + obj.Epsilon);
                    end
                    % 2) Batch normalization operation further shifts and scales the activations using Scale and Offset values
                    % 2a) Scale values
                    for i=1:obj.NumChannels
                        x.V(:,:,:,i,:) = x.V(:,:,:,i,:) .* obj.Scale(:,:,:,i);
                    end
                    % 2b) Apply offset
                    for i=1:obj.NumChannels
                        x.V(:,:,:,i,1) = x.V(:,:,:,i,1) + obj.Offset(:,:,:,i);
                    end
                    image = x;
                end
            
            elseif isa(image, 'ImageStar')
                % Get parameters to the right shape
                if(length(size(obj.TrainedMean)) == 2) && size(image.V, 1) == 1 && size(image.V, 2) == 1 && length(size(image.V)) == 4
                    obj.TrainedMean = reshape(obj.TrainedMean, [1 1 size(obj.TrainedMean, 1)]);
                    obj.TrainedVariance = reshape(obj.TrainedVariance, [1 1 size(obj.TrainedVariance, 1)]);
                    obj.Epsilon = reshape(obj.Epsilon, [1 1 size(obj.Epsilon, 1)]);
                    
                    if length(size(obj.Offset)) ~= 3
                        obj.Offset = reshape(obj.Offset, [1 1 size(obj.Offset, 1)]);
                    end
                    
                    if length(size(obj.Scale)) ~= 3
                        obj.Scale = reshape(obj.Scale, [1 1 size(obj.Scale, 1)]);
                    end
                    obj.NumChannels = 1;
                end
                
                % Begin reachability analysis
                x = in_image;
                
                % If mean and variance exist
                if ~isempty(obj.TrainedMean) && ~isempty(obj.TrainedVariance) && ~isempty(obj.Epsilon) && ~isempty(obj.Offset) && ~isempty(obj.Scale)
                    % 1) Normalized all elements of x (for each channel independently)
                    % 1a) substract mean from elements of x per channel
                    for i=1:obj.NumChannels
                        x.V(:,:,i,1) = x.V(:,:,i,1) - obj.TrainedMean(:,:,i);
                    end
                    % 1b) Divide by sqrt(variance + epsilon)
                    for i=1:obj.NumChannels
                        x.V(:,:,i,:) = x.V(:,:,i,:) .* 1/sqrt(obj.TrainedVariance(:,:,i) + obj.Epsilon);
                    end
                    % 2) Batch normalization operation further shifts and scales the activations using Scale and Offset values
                    % 2a) Scale values
                    for i=1:obj.NumChannels
                        x.V(:,:,i,:) = x.V(:,:,i,:) .* obj.Scale(:,:,i);
                    end
                    % 2b) Apply offset
                    for i=1:obj.NumChannels
                        x.V(:,:,i,1) = x.V(:,:,i,1) + obj.Offset(:,:,i);
                    end
                    image = x;
                end
                
            elseif isa(image, 'Star')
                l = obj.Scale ./ sqrt(obj.TrainedVariance + obj.Epsilon);
                for i=1:size(image.V, 2)
                    image.V(:, i) = ((image.V(:, i) - obj.TrainedMean) .* l + obj.Offset);
                end
            end
            
        end

        % Reachability analysis when in_image is an array of ImageStar sets 
        function images = reach_star_multipleInputs(obj, in_images, option)
            % @in_images: an array of ImageStars input set
            % @option: = 'parallel' or 'single' or empty
            
            % author: Dung Tran
            % date: 1/7/2020
            
            n = length(in_images);
            if isa(in_images(n), 'VolumeStar')
                images(n) = VolumeStar;
            elseif isa(in_images(n), 'ImageStar')
                images(n) = ImageStar; 
            else
                images(n) = Star; 
            end 
            
            if strcmp(option, 'parallel')
                parfor i=1:n
                    images(i) = obj.reach_star_single_input(in_images(i));
                end
            elseif strcmp(option, 'single') || isempty(option)
                for i=1:n
                    images(i) = obj.reach_star_single_input(in_images(i));
                end
            else
                error('Unknown computation option');

            end
        end
        
        % Main reach function
        function images = reach(varargin)
            % @in_image: an input imagestar or imagezono
            % @image: output set
            % @option: = 'single' or 'parallel' 
            
            % author: Dung Tran
            % date: 6/26/2019
             
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
                    option = varargin{4};
                
                case 3
                    obj = varargin{1};
                    in_images = varargin{2};
                    method = varargin{3};
                    option = [];
                otherwise
                    error('Invalid number of input arguments (should be 2, 3, 4, 5, or 6)');
            end
            
            
            if strcmp(method, 'approx-star') || strcmp(method, 'exact-star') || strcmp(method, 'abs-dom')|| contains(method, "relax-star")
                images = obj.reach_star_multipleInputs(in_images, option);
            else
                error("Unknown reachability method");
            end
          
        end
        
    end
    
    
    methods(Static)

        % parse a trained batch normalization layer from matlab
        function L = parse(layer)
            % @layer: batch normalization layer
            % @L: constructed layer
            
            if ~isa(layer, 'nnet.cnn.layer.BatchNormalizationLayer')
                error('Input is not a Matlab nnet.cnn.layer.BatchNormalizationLayer class');
            end
            L = BatchNormalizationLayer('Name', layer.Name, 'NumChannels', layer.NumChannels, 'TrainedMean', layer.TrainedMean, 'TrainedVariance', layer.TrainedVariance, 'Epsilon', layer.Epsilon, 'Offset', layer.Offset, 'Scale', layer.Scale, 'NumInputs', layer.NumInputs, 'InputNames', layer.InputNames, 'NumOutputs', layer.NumOutputs, 'OutputNames', layer.OutputNames);
            
        end
        
    end
    
end

