classdef ReluLayer < ActivationFunctionLayer
    % The Relu layer class in CNN
    %   Contain constructor and reachability analysis methods
    % Main references:
    % 1) An intuitive explanation of convolutional neural networks: 
    %    https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
    % 2) More detail about mathematical background of CNN
    %    http://cs231n.github.io/convolutional-networks/
    %    http://cs231n.github.io/convolutional-networks/#pool
    % 3) Matlab implementation of Convolution2DLayer and MaxPooling (for training and evaluating purpose)
    %    https://www.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html
    %    https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.relulayer.html
    
    %   Dung Tran: 6/20/2019
    %   update:
    %       Diego Manzanas: 12/07/2022
    %        - Inheritance using ActivationFunctionLayer
    
    % setting hyperparameters method
    methods
        
        % constructor of the class
        function obj = ReluLayer(varargin)           
            % author: Dung Tran
            % date: 6/26/2019    
            % update: Diego Manzanas, 12/07/2022
            obj = obj@ActivationFunctionLayer(varargin);
        end
        
        
    end
        
    
    methods % evaluation methods
        
        function y = evaluate(~, input)
            % @input: 2 or 3-dimensional array, for example, input(:, :, :), 
            % @y: 2 or 3-dimensional array, for example, y(:, :, :)
            
            % author: Dung Tran
            % date: 6/26/2019
            
            % @y: high-dimensional array (output volume)

            n = size(input);
            N = prod(n);
            
            I = reshape(input, [N 1]);
            y = PosLin.evaluate(I);
            y = reshape(y, n);

        end
        
    end
        
    
    methods % reachability methods

        % reachability using ImageStar
        function images = reach_star_single_input(~, in_image, method, relaxFactor, dis_opt, lp_solver)
            % @in_image: an ImageStar input set
            % @method: = 'exact-star' or 'approx-star' or 'abs-dom'
            % @relaxFactor: of approx-star method
            % @dis_opt: display option = [] or 'display'
            % @lp_solver: lp solver
            % @images: an array of ImageStar (if we use 'exact-star' method)
            %         or a single ImageStar set
                        
            if ~isa(in_image, 'VolumeStar') && ~isa(in_image, 'ImageStar') && ~isa(in_image, 'Star')
                error('input is not an ImageStar, Star or VolumeStar');
            end
            if isa(in_image, 'VolumeStar')
                % get dimensions of VolumeStar
                h = in_image.height;
                w = in_image.width;
                dp = in_image.depth;
                c = in_image.numChannel;
                % transform to star and compute relu reachability
                Y = PosLin.reach(in_image.toStar, method, [], relaxFactor, dis_opt, lp_solver); % reachable set computation with ReLU
                n = length(Y);
                % transform back to VolumeStar
                images(n) = VolumeStar;
                for i=1:n
                    images(i) = Y(i).toVolumeStar(h,w,dp, c);
                end
            elseif isa(in_image, 'ImageStar')
                h = in_image.height;
                w = in_image.width;
                c = in_image.numChannel;
                reduceMem = 0;
                if contains(method, "reduceMem")
                    method = split(method, '-');
                    method = strjoin(method(1:end-1),'-');
                    reduceMem = 1;
                end
                Y = PosLin.reach(in_image.toStar, method, [], relaxFactor, dis_opt, lp_solver); % reachable set computation with ReLU
                n = length(Y);
                images(n) = ImageStar;
                % transform back to ImageStar
                for i=1:n
                    images(i) = Y(i).toImageStar(h,w,c);
                    if reduceMem
                        [l,u] = images(i).estimateRanges;
                        % Get dimensions
                        h = images(i).height;
                        w = images(i).width;
                        c = images(i).numChannel;
                        % create ImageStar variables
                        center = cast(0.5*(u+l), 'like', in_image.V);
                        v = 0.5*(u-l);
                        idxRegion = find((l-u));
                        n = length(idxRegion);
                        V = zeros(h, w, c, n, 'like', center);
                        bCount = 1;
                        for i1 = 1:size(v,1)
                            for i2 = 1:size(v,2)
                                basisValue = v(i1,i2);
                                if basisValue
                                    V(i1,i2,:,bCount) = v(i1,i2);
                                    bCount = bCount + 1;
                                end
                            end
                        end
                        C = zeros(1, n, 'like', V);
                        d = zeros(1, 1, 'like', V);
                        % Construct ImageStar
                        images(i) = ImageStar(cat(4,center,V), C, d, -1*ones(n,1), ones(n,1), l, u);
                    end
                end
            else % star
                images = PosLin.reach(in_image, method, [], relaxFactor, dis_opt, lp_solver); % reachable set computation with ReLU
            end

        end

    end
    
    
    methods(Static)
         % parse a trained relu Layer from matlab
        function L = parse(relu_layer)
            % @relu_layer: a average pooling 2d layer from matlab deep
            % neural network tool box
                        
            % author: Dung Tran
            % date: 6/17/2019
            
            if ~isa(relu_layer, 'nnet.cnn.layer.ReLULayer')
                error('Input is not a Matlab nnet.cnn.layer.ReLULayer class');
            end
            
            L = ReluLayer(relu_layer.Name, relu_layer.NumInputs, relu_layer.InputNames, relu_layer.NumOutputs, relu_layer.OutputNames);
            
        end
        
    end
    
end

