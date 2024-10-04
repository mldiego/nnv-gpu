classdef NN < handle
    % NN class encodes all type of neural networks supported in NNV
        %   This class is part of the refactoring plan for NNV executed at the
        %   end of 2022. This class generalizes previous constructors for
        %   neuralODEs, FFNNS, CNN, BNN, RNN,...
        %   In addition, we will add a property Connections, in order to support
        %   other type of DL models like residual or U networks.

    % Reachability analysis methods are developed for each individual layer
    
    % Author: Diego Manzanas Lopez
    % Date: 09/28/2022
    % Notes: Code is based on the previous CNN and FFNNS classes written by
    %             Dr. Hoang Dung Tran
    % This is a generalized class, created in the refactoring of NNV in 2023 (NNV 2.0)
    %    It supports FFNNS, CNN, SEGNET, BNN and RNN from previous NNV version
    
    properties
        
        Name = 'nn'; % name of the network
        Layers = {}; % An array of Layers, eg, Layers = [L1 L2 ...Ln]
        Connections = []; % A table specifying source and destination layers
        numLayers = 0; % number of Layers
        numNeurons = 0; % number of Neurons
        InputSize = 0; % number of Inputs
        OutputSize = 0; % number of Outputs
        
        % properties for reach set  and evaluation computation
        reachMethod = 'approx-star';    % reachable set computation scheme, default - 'approx-star'
        relaxFactor = 0; % default - solve 100% LP optimization for finding bounds in 'approx-star' method
        reachOption = []; % parallel option, default - non-parallel computing
        numCores = 1; % number of cores (workers) using in computation
        reachSet = {};  % reachable set for each layers
        reachTime = []; % computation time for each layers
        features = {}; % outputs of each layer in an evaluation
        input_vals = {}; % input values to each layer (cell array of cells of layer input values)
        input_sets = {}; % input set values for each layer (cell array of cells of layer input sets)
        dis_opt = []; % display option = 'display' or []
        lp_solver = 'linprog'; % choose linprog as default LP solver for constructing reachable set user can choose 'glpk' or 'linprog' as an LP solver
        
        % To facilitate graph computation flow
        name2indx = []; % Match name to index in nnvLayers list
    end


    methods % main methods (constructor, evaluation, reach)
        
        % constructor
        function obj = NN(varargin)
            
            switch nargin
                % if connections undefined, assume NN is fullyconnected
                % No skipped connections, no multiple connections from any layer
                case 5
                    % parse inputs
                    Layers = varargin{1};
                    conns = varargin{2}; % connections
                    inputSize = varargin{3};
                    outputSize = varargin{4};
                    name = varargin{5};
                    nL = length(Layers); % number of Layers

                case 4
                    % parse inputs 
                    Layers = varargin{1};
                    if isa(varargin{2}, 'table')
                        conns = varargin{2};
                        inputSize = varargin{3};
                        outputSize = varargin{4};
                        name = 'nn';
                    else
                        conns = [];
                        inputSize = varargin{2};
                        outputSize = varargin{3};
                        name = varargin{4};
                    end
                    nL = length(Layers); % number of Layers

                case 2 % only layers and connections defined
                    Layers = varargin{1};
                    conns = varargin{2};
                    nL = length(Layers);
                    name = 'nn';
                    inputSize = 0;
                    outputSize = 0;

                case 1 % only layers, assume each layer is only connected to the next one
                    Layers = varargin{1};
                    nL = length(Layers);
                    conns = [];
                    name = 'nn';
                    inputSize = 0;
                    outputSize = 0;

                case 0
                    name = 'nn';
                    Layers = {};
                    conns = [];
                    nL = 0;
                    inputSize = 0;
                    outputSize = 0;

                otherwise
                    error('Invalid number of inputs, should be 0, 1, 2, 3 or 5');
            end

            % update object properties
            obj.Name = name;              % Name of the network
            obj.Layers = Layers;          % Layers in NN
            obj.Connections = conns;       % Connections in NN
            obj.numLayers = nL;           % number of layers
            obj.InputSize = inputSize;    % input size
            obj.OutputSize = outputSize;  % output size
                      
        end
        
        % Evaluation of a NN (test it, fix for neuralode with multiple outputs)
        function y = evaluate(obj, x)
            % Evaluate NN given an input sample
            % y = NN.evaluate(x)
            % @x: input vector x
            % @y: output vector y
            
            % Two options to exectute evaluation
            % 1) Connections are defined
            if ~isempty(obj.Connections)
                 y = obj.evaluate_withConns(x);
            % 2) No connections defined, execute each layer consecutively
            else
                y = obj.evaluate_noConns(x);
            end

        end
        
        % evaluate parallel
        function y = evaluate_parallel(obj, inputs)
            % @inputs: array of inputs
            % @y: output vector
            
            n = length(inputs);
            y = cell(1, n);
            
            if obj.numCores < 1
                error("Invalid number of Cores");
            elseif obj.numCores == 1
                for i=1:n
                    y{i} = obj.evaluate(inputs{i});
                end
            else
                obj.start_pool();
                parfor i=1:n
                    y{i} = obj.evaluate(inputs{i});
                end
            end
        end

        % Define the reachability function for any general NN (test it)
        function outputSet = reach(obj, inputSet, reachOptions)
            % inputSet: input set (type -> Star, ImageStar, Zono or ImageZono)   
            % reachOptions: reachability options for NN (type -> struct)
            %       Required Fields:
            %           reachMethod (string)
            %               select from {'approx-star','exact-star', 'abs-dom', approx-zono'}
            %       Optional Fields:
            %           numCores (int) -> default = 1
            %           relaxFactor (int) -> default = 0
            %           dis_opt ('display' or []), display options, use for debugging, default = [] (no display)
            %           lp_solver ('glpk', 'linprog') -> default = 'glpk'

            % Parse the inputs

            % Ensure input set is a valid type
            if ~isa(inputSet,"Star") && ~isa(inputSet,"ImageStar") && ~isa(inputSet, "VolumeStar")...
                    && ~isa(inputSet,"ImageZono") && ~isa(inputSet,"Zono") 
                error('Wrong input set type. Input set must be of type "Star", "ImageStar", "VolumeStar", "ImageZono", or "Zono"')
            end

            % Check validity of reachability method
            if exist("reachOptions",'var')
                reachOptions = validate_reach_options(obj, reachOptions);
            else
                reachOptions = struct; % empty options, run with default values
            end

            % Change if want to execute on GPU
            if isfield(reachOptions, 'device')
                if strcmp(reachOptions.device, 'gpu')
                    obj = obj.params2gpu;
                    inputSet = inputSet.changeDevice('gpu');
                end
            end

            % Process reachability options
            if ~isstruct(reachOptions)
                error("The reachability parameters must be specified as a struct.")
            else
                if isfield(reachOptions, 'reachMethod')
                    obj.reachMethod = reachOptions.reachMethod;
                end
                if isfield(reachOptions, 'numCores')
                    obj.numCores = reachOptions.numCores;
                else
                    obj.numCores = 1;
                end
                if isfield(reachOptions, 'relaxFactor')
                    obj.relaxFactor = reachOptions.relaxFactor;
                end
                if isfield(reachOptions, 'dis_opt')
                    obj.dis_opt = reachOptions.dis_opt; % use for debuging
                end
                if isfield(reachOptions, 'lp_solver')
                    obj.lp_solver = reachOptions.lp_solver;
                end
            end
            
            % Parallel computation or single core?
            if  obj.numCores > 1
                obj.start_pool;
                obj.reachOption = 'parallel';
            end

            % Debugging option
            if strcmp(obj.dis_opt, 'display')
                fprintf('\nPerform reachability analysis for the network %s \n', obj.Name);
            end

            % ensure NN parameters and input set share same precision
            inputSet = obj.consistentPrecision(inputSet); % change only input, this can be changed in the future
            
            % Perform reachability based on connections or assume no skip/sparse connections
            if isempty(obj.Connections)
                outputSet = obj.reach_noConns(inputSet);
            else 
                outputSet = obj.reach_withConns(inputSet);
            end

        end
    end
     
    
    methods % secondary methods (verification, safety, robustness...)
        
        % Verify a VNN-LIB specification
        function result = verify_vnnlib(obj, propertyFile, reachOptions)
            
            % Load specification to verify
            property = load_vnnlib(propertyFile);
            lb = property.lb;
            ub = property.ub;

            % Create reachability parameters and options
            if contains(reachOptions.reachMethod, "zono")
                X = ImageZono(lb, ub);
            else
                X = ImageStar(lb,ub);
            end

            % Compute reachability
            Y = obj.reach(X, reachOptions); % Seems to be working
            result = verify_specification(Y, property.prop); 

            % Modify in case of unknown and exact
            if result == 2
                if contains(obj.reachMethod, "exact")
                    result = 0;
                end
            end
    
        end
        
        % Check robustness of output set given a target label
        function rb = checkRobust(obj, outputSet, target)
            % rb = checkRobust(~, outputSet, target)
            % 
            % @outputSet: the outputSet we need to check
            %
            % @target: the correct_id of the classified output or a halfspace defining the robustness specification
            % @rb: = 1 -> robust
            %      = 0 -> not robust
            %      = 2 -> unknown
            
            % Process set
            if ~isa(outputSet, "Star")
                nr = length(outputSet);
                R = Star;
                for s=1:nr
                    R(s) = outputSet(s).toStar;
                end
            else
                R = outputSet;
            end
            % Process robustness spec
            if ~isa(target, "HalfSpace")
                if isscalar(target)
                    target = obj.robustness_set(target, 'max');
                else
                    error("Target must be a HalfSpace (unsafe/not robust region) or as a scalar determining the output label target.")
                end
            end
            % Check robustness
            rb = verify_specification(R, target);

        end % end check robust
        
        % Verify robutness of a NN given an input set and target (id or HalfSpace)
        function result = verify_robustness(obj, inputSet, reachOptions, target)
            % Compute reachable set
            R = obj.reach(inputSet, reachOptions);
            % Check robustness
            result = obj.checkRobust(R, target);
            % Modify in case of unknown and exact
            if result == 2
                if contains(obj.reachMethod, "exact")
                    result = 0;
                end
            end
        end
        
        % Classify input set (one or more label_id?)
        function label_id = classify(obj, inputValue, reachOptions)
            % label_id = classify(inputValue, reachOptions = 'default')
            % inputs
                % @inputValue: input to the NN, typically an image, or an imagestar 
                % reachOptions: 'default' means no reach parameters are
                    % chosen, to be used also when input is not a set
            % @label_id: output index of classified object
            % Assume the network outputs the largest output value

            % parse inputs
            switch nargin
                case 1
                    reachOptions = 'default';
                case 2
                otherwise
                    error('Invalid number of inputs, should be 1 or 2');
            end
            
            % Check if input is not a set, then proceed with classification
            if ~isa(inputValue, 'ImageStar') && ~isa(inputValue, 'Star') && ~isa(inputValue, 'Zono') && ~isa(inputValue, 'ImageZono')
                y = obj.evaluate(in_image);
                y = reshape(y, [obj.OutputSize, 1]);
                [~, label_id] = max(y); 
            else
                % For input sets, compute reach set and then classify
                if ~isstruct(reachOptions)
                    reachOptions.method = 'approx-star'; % default reachability method
                    reachOptions.numOfCores = 1;
                end
                RS = obj.reach(inputValue, reachOptions);
                % For now, we are converting the set to ImageStar and then
                % computing max values for classification
                n = length(RS);
                label_id = cell(n, 1);
                for i=1:n
                    rs = RS(i);
                    if isa(rs, 'Star') || isa(rs, 'Zono')
                        rs = rs.toImageStar(rs.dim, 1, 1);
                    elseif isa(rs, 'ImageZono')
                        rs = rs.toImageStar;
                    end
                    new_rs  = ImageStar.reshape(rs, [obj.OutputSize(1) 1 1]);                    
                    max_id = new_rs.get_localMax_index([1 1], [obj.OutputSize(1) 1], 1);
                    label_id{i} = max_id(:, 1);
                end
            end
        end
    end
    

    % helper functions
    methods
        
        % Check reachability options defined are allowed
        function reachOptions = validate_reach_options(obj, reachOptions)
            reach_method = reachOptions.reachMethod;
            if contains(reach_method, "exact")
                for i=1:length(obj.Layers)
                    if isa(obj.Layers{i}, "ODEblockLayer")
                        if ~isa(obj.Layers{i}.odemodel,'LinearODE')
                            warning("Exact reachability is not possible with a neural ODE layer (" + class(obj.Layers{i}.odemodel) + "). Switching to approx-star.");
                            reachOptions.reachMethod = "approx-star";
                        end
                    end
                    if isa(obj.Layers{i}, "SigmoidLayer") || isa(obj.Layers{i}, "TanhLayer")
                        warning("Exact reachability is not possible with layer " + class(obj.Layers{i}) + ". Switching to approx-star.");
                        reachOptions.reachMethod = "approx-star";
                    end
                end
            elseif contains(reach_method, "relax-star")
                if ~isfield(reachOptions, "relaxFactor")
                    error("Please, specify a relax factor value to perform relax-star reachability analysis")
                else
                    if reachOptions.relaxFactor < 0 || reachOptions.relaxFactor > 1
                        error('Invalid relaxFactor. The value of relax factor must be between 0 and 1');
                    end
                end
            end
            % Ensure reachability is done in a single core
            if ~contains(reach_method, 'exact')
                reachOptions.reachOption = 'single';
                reachOptions.numCores = 1;
            end
            
        end
        
        % Ensure input and parameter precision is the same
        function inputSet = consistentPrecision(obj, inputSet)
            % (assume parameters have same precision across layers)
            % approach: change input precision based on network parameters
            inputPrecision = class(inputSet(1).V);
            netPrecision = 'double'; % default
            for i=1:length(obj.Layers)
                if isa(obj.Layers{i}, "FullyConnectedLayer") || isa(obj.Layers{i}, "Conv2DLayer")
                    netPrecision = class(obj.Layers{i}.Weights);
                    break;
                end
            end
            if ~strcmp(inputPrecision, netPrecision)
                % input and parameter precision does not match
                warning("Changing input set precision to "+string(netPrecision));
                for i = 1:length(inputSet)
                    inputSet(i) = inputSet(i).changeVarsPrecision(netPrecision);
                end
            end
        end

        % Change paramters to gpu
        function obj = params2gpu(obj)
            % change the parameters layer by layer
            for i = 1:length(obj.Layers)
                gpuLayer = obj.Layers{i}.toGPU;
                obj.Layers{i} = gpuLayer;
            end
        end

        % Change params precision
        function obj = changeParamsPrecision(obj, precision)
            % change the parameters layer by layer
            for i = 1:length(obj.Layers)
                pLayer = obj.Layers{i}.changeParamsPrecision(precision);
                obj.Layers{i} = pLayer;
            end
        end

        % Create input set based on input vector and bounds
        function R = create_input_set(obj, x_in, disturbance, lb_allowable, ub_allowable) % assume tol is applied to every vale of the input
            % R = create_input_set(obj, x_in, disturbance, lb_allowable, ub_allowable)
            % @R = Star or ImageStar

            lb = x_in;
            ub = x_in;
            n = numel(x_in); % number of elements in array
            % Apply disturbance
            lb = lb - disturbance;
            ub = ub + disturbance;
            % Check is disturbance value is allowed (lb)
            if ~isempty(lb_allowable)
                for i=1:n
                    if lb(i) < lb_allowable(i)
                        lb(i) = lb_allowable(i);
                    end
                end
            end
            % Check is disturbance value is allowed (ub)
            if ~isempty(ub_allowable)
                for i=1:n
                    if ub(i) > ub_allowable(i)
                        ub(i) = ub_allowable(i);
                    end
                end
            end
            
            R = obj.set_or_imageset(lb, ub);

        end
        
        % Given input bounds, create input set
        function R = set_or_imageset(obj, lb, ub)
            if length(ub) ~= numel(ub) % not a vector, so ImageStar
                if contains(obj.reachMethod, 'zono')
                    R = ImageZono(lb,ub);
                else
                    R = ImageStar(lb,ub);
                end
            else
                if contains(obj.reachMethod, 'zono')
                    R = Zono(lb,ub);
                else
                    R = Star(lb,ub);
                end
                for k=1:length(obj.Layers)
                    if isa(obj.Layers{k}, 'ImageInputLayer') || isa(obj.Layers{k}, "Conv2DLayer") || contains(class(obj.Layers{k}), "Pooling")
                        if contains(obj.reachMethod, 'zono')
                            R = ImageZono(lb,ub);
                        else
                            R = ImageStar(lb,ub);
                        end
                        break;
                    end
                end
            end
        end

        % Create unsafe/not robust region from a target label of a classification NN
        function Hs = robustness_set(obj, target, class_type)
            % @Hs: unsafe/not robust region defined as a HalfSpace
            %  - target: label idx of the given input set
            %  - class_type: assume max, but could also be min like in ACAS Xu ('min', 'max')

            outSize = obj.OutputSize;
            if outSize == 0 % output size was not properly defined when creating the NN
                layer = obj.Layers{end};
                if isa(layer, "FullyConnectedLayer")
                    outSize = length(layer.Bias);
                else
                    error("Output size is set to 0, but it must be >= 1");
                end
            elseif target > outSize
                error("Target idx must be less than or equal to the output size of the NN.");
            end

            % Define HalfSpace Matrix and vector
            G = ones(outSize,1);
            G = diag(G);
            G(target, :) = [];
            if strcmp(class_type, "max")
                G = -G;
                G(:, target) = 1;
            elseif strcmp(class_type, "min")
                G(:, target) = -1;
            end
%             g = zeros(height(G),1);

            % Create HalfSapce to define robustness specification
            Hs = [];
            for i=1:height(G)
                Hs = [Hs; HalfSpace(G(i,:), 0)];
            end
        end

        % start parallel pool for computing 
        function start_pool(obj)

            if obj.numCores > 1
                poolobj = gcp('nocreate'); % If no pool, do not create new one.
                if isempty(poolobj)
                    parpool('local', obj.numCores); 
                else
                    if poolobj.NumWorkers ~= obj.numCores
                        delete(poolobj); % delete the old poolobj
                        parpool('local', obj.numCores); % start the new one with new number of cores
                    end                    
                end
            end   
        end
        
        % evaluate NN when no connections are defined
        function y = evaluate_noConns(obj, x)
            y = x;
            % reset eval related parameters
            obj.features = cell(1, length(obj.Layers));
            for i=1:obj.numLayers
                y = obj.Layers{i}.evaluate(y);
                obj.features{i} = y;
            end
        end
        
        % evaluate NN based on connections table (test it)
        function y = evaluate_withConns(obj, x)
            % rest eval values
            obj.features = cell(1, obj.numLayers); % store output for each layer
            obj.input_vals = cell(1, obj.numLayers); % store input for each layer
           
            % Evaluate layer-by-layer based on connection graph
            for i=1:height(obj.Connections)
                % 1) Get name and index of layer
                source = obj.Connections.Source(i);
                % ensure we get just the name and not specific properties
                source = split(source, '/');
                source = source{1};
                source_indx = obj.name2indx(source); % indx in Layers array
                
                % 2) Get input to layer
                if i > 1
                    x = obj.input_vals{source_indx}; % get inputs to layer unless it's the first layer
                else
                    obj.input_vals{1} = x;
                end
                
                % 3) evaluate layer
                exec_len = length(obj.features);
                % ensure layer has not been evaluated yet
                if isempty(obj.features{source_indx})
                    if isa(obj.Layers{source_indx}, 'MaxUnpooling2DLayer')
                        obj.Layers{source_indx}.MaxPoolIndx = obj.Layers{obj.name2indx(obj.Layers{source_indx}.PairedMaxPoolingName)}.MaxIndx;
                        obj.Layers{source_indx}.MaxPoolSize = obj.Layers{obj.name2indx(obj.Layers{source_indx}.PairedMaxPoolingName)}.InputSize;
                    end
                    y = obj.Layers{source_indx}.evaluate(x);
                    obj.features{source_indx} = y;
                else

                end
                
                % 4) save inputs to destination layer
                dest = obj.Connections.Destination(i);
                dest = split(dest, '/');
                dest_name = dest{1};
                dest_indx = obj.name2indx(dest_name); % indx in Layers array
                % check if there source has multiple inputs (concat layer, unpooling ...)
                if length(dest) > 2
                    % unpooling option
                    if isa(obj.Layers{dest_indx}, 'MaxUnpooling2DLayer')
                        destP = dest{2};
                        if strcmp(destP, 'in')
                            obj.input_vals{dest_indx} = y; % store layer input
                        else
                            error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                        end
                    % concatenation layers (2 or more inputs with specific order {in1, in2, ... inX})
                    elseif contains(class(obj.Layers{dest_indx}), 'Concatenation')
                        destP = dest{2};
                        if startsWith(destP, 'in')
                            input_num = str2double(destP(3:end));
                        else
                            error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                        end
                        obj.input_vals{dest_indx}{input_num} = y;
                    % concatenation layers (2 or more inputs with specific order {in1, in2, ... inX})
                    elseif isa(obj.Layers{dest_indx}, 'DepthConcatenationLayer')
                        destP = dest{2};
                        if startsWith(destP, 'in')
                            input_num = str2double(destP(3:end));
                        else
                            error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                        end
                        obj.input_vals{dest_indx}{input_num} = y;
                    elseif isa(obj.Layers{dest_indx}, 'AdditionLayer')
                        destP = dest{2};
                        if startsWith(destP, 'in')
                            input_num = str2double(destP(3:end));
                        else
                            error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                        end
                        obj.input_vals{dest_indx}{input_num} = y;
                    else
                        error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                    end
                else
                    obj.input_vals{dest_indx} = y;
                end
            end
                
            % Check if last layer is executed (default is last layer is not executed, but in the 
            % case of the PixelClassificationLayer this is necessary)
            % Assume last layer in array is the output layer
            if isempty(obj.features{end})
                x = obj.input_vals{end};
                y = obj.Layers{end}.evaluate(x);
                obj.features{end} = y;
            end
        end
        
        % reach NN when no connections are defined (test it)
        function outSet = reach_noConns(obj, inSet)
            % Initialize variables
            obj.reachSet = cell(1, obj.numLayers);
            obj.reachTime = zeros(1, obj.numLayers);
            if strcmp(obj.dis_opt, 'display')
                fprintf('\nPerform reachability analysis for the network %s...', obj.Name);
            end
            % Begin reach set computation
            rs = inSet;
            for i=2:obj.numLayers+1
                if strcmp(obj.dis_opt, 'display')
                    fprintf('\nPerforming analysis for Layer %d (%s)...', i-1, obj.Layers{i-1}.Name);
                end
                start_time = tic;
                rs_new = obj.Layers{i-1}.reach(rs, obj.reachMethod, obj.reachOption, obj.relaxFactor, obj.dis_opt, obj.lp_solver);
                obj.reachTime(i-1) = toc(start_time); % track reach time for each layer
                rs = rs_new; % get input set to next layer
                obj.reachSet{i-1} = rs_new; % store output set for layer
            end
            % Output
            outSet = rs_new;
        end

        % reach NN based on connections table (test it)
        function outSet = reach_withConns(obj, inSet, varargin)
            % Initialize variables to store reachable sets and computation time
            obj.reachSet = cell(1, obj.numLayers);
            obj.reachTime = zeros(1, obj.numLayers);
            obj.input_sets = cell(1, height(obj.Connections)); % store input reach sets for each layer
            if strcmp(obj.dis_opt, 'display')
                fprintf('\nPerform reachability analysis for the network %s...', obj.Name);
            end

            if nargin == 3
                reachType = varargin{1};
            else
                reachType = 'default';
            end

            % Begin reachability computation
            for i=1:height(obj.Connections)

                % 1) Get name and index of layer
                source = obj.Connections.Source(i);
                % ensure we get just the name and not specific properties
                source = split(source, '/');
                source = source{1};
%                 fprintf('\n layer reachability: %s', source);
                source_indx = obj.name2indx(source); % indx in Layers array
                
                % 2) Get input to layer
                if i > 1
                    inSet = obj.input_sets{source_indx}; % get inputs to layer unless it's the first layer
                end
                
                % 3) reach layer
                exec_len = length(obj.reachSet);
                % ensure layer has not been eexcuted yet
                if exec_len >= source_indx && isempty(obj.reachSet{source_indx})
                    if strcmp(obj.dis_opt, 'display')
                        fprintf('\nPerforming analysis for Layer %d (%s)...', i-1, source);
                    end
                    t = tic;
                    if strcmp(reachType, 'sequence')
                        outSet = obj.Layers{source_indx}.reachSequence(inSet, obj.reachMethod, obj.reachOption, obj.relaxFactor, obj.dis_opt, obj.lp_solver);
                    else
                        outSet = obj.Layers{source_indx}.reach(inSet, obj.reachMethod, obj.reachOption, obj.relaxFactor, obj.dis_opt, obj.lp_solver);
                    end
                    obj.reachTime(source_indx) = toc(t);
                    obj.reachSet{source_indx} = outSet;
                end
                
                % 4) save inputs to destination layer
                dest = obj.Connections.Destination(i);
                dest = split(dest, '/');
                dest_name = dest{1};
                dest_indx = obj.name2indx(dest_name); % indx in Layers array
                % check if there source has multiple inputs (concat layer, unpooling ...)
                if length(dest) > 1
                    % unpooling option
                    if isa(obj.Layers{dest_indx}, 'MaxUnpooling2DLayer')
                        destP = dest{2};
                        if strcmp(destP, 'in')
                            obj.input_sets{dest_indx} = outSet; % store layer input
                        elseif strcmp(destP, 'indices')
                            obj.Layers{dest_indx}.MaxPoolIndx = obj.Layers{source_indx}.MaxIndx;
                        elseif strcmp(destP, 'size')
                            obj.Layers{dest_indx}.MaxPoolSize = obj.Layers{source_indx}.InputSize;
                        else
                            error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                        end
                    % concatenation layers (2 or more inputs with specific order {in1, in2, ... inX})
                    elseif contains(class(obj.Layers{dest_indx}), 'Concatenation')
                        destP = dest{2};
                        if startsWith(destP, 'in')
                            input_num = str2double(destP(3:end));
                        else
                            error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                        end
                        obj.input_sets{dest_indx}{input_num} = outSet;
                    % concatenation layers (2 or more inputs with specific order {in1, in2, ... inX})
                    elseif isa(obj.Layers{dest_indx}, 'DepthConcatenationLayer')
                        destP = dest{2};
                        if startsWith(destP, 'in')
                            input_num = str2double(destP(3:end));
                        else
                            error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                        end
                        obj.input_sets{dest_indx}{input_num} = outSet;
                    elseif isa(obj.Layers{dest_indx}, 'AdditionLayer')
                        destP = dest{2};
                        if startsWith(destP, 'in')
                            input_num = str2double(destP(3:end));
                        else
                            error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                        end
                        obj.input_sets{dest_indx}{input_num} = outSet;
                    else
                        error("Destination not valid ("+string(obj.Connections.Destination(i))+")");
                    end
                else
                    obj.input_sets{dest_indx} = outSet;
                end
            end
            % Check if last layer is executed (default is last layer is not executed, but in the 
            % case of the PixelClassificationLayer this is necessary)
            % Assume last layer in array is the output layer
            if isempty(obj.reachSet{end})
                inSet = obj.reachSet{end-1};
                if strcmp(reachType, 'sequence')
                    outSet = obj.Layers{end}.reachSequence(inSet, obj.reachMethod, obj.reachOption, obj.relaxFactor, obj.dis_opt, obj.lp_solver);
                else
                    outSet = obj.Layers{end}.reach(inSet, obj.reachMethod, obj.reachOption, obj.relaxFactor, obj.dis_opt, obj.lp_solver);
                end
                obj.reachSet{end} = outSet;
            end
        end
        
    end % end helper functions
    
end

