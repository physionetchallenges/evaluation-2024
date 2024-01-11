% This file contains functions for evaluating models for the 2024 Challenge. You can run it as follows:
%
%   evaluate_model(labels, outputs, scores.csv)
%
% where 'labels' is a folder containing files with the labels, 'outputs' is a folder containing files with the outputs from your
% model(s), and 'scores.csv' (optional) is a collection of scores for the model outputs.
%
% Each label or output file must have the format described on the Challenge webpage.

function evaluate_model(labels, outputs, scores)
    % Check for Python and NumPy.
    command = 'python -V';
    [status, ~] = system(command);
    if status~=0
        error('Python not found: please install Python or make it available as the command "python ...".');
    end

    command = 'python -c "import numpy"';
    [status, ~] = system(command);
    if status~=0
        error('NumPy not found: please install NumPy or make it available to Python.');
    end

    % Define command for evaluating model outputs.
    switch nargin
        case 2
            command = ['python evaluate_model.py' ' -d ' labels ' -o ' outputs];
        case 3
            command = ['python evaluate_model.py' ' -d ' labels ' -o ' outputs ' -s ' scores];
        otherwise
            command = '';
    end

    % Evaluate model outputs.
    [~, output] = system(command);
    fprintf(output);
end
