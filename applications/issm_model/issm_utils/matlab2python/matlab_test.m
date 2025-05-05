function [x, y] = matlab_test(a, b)

    disp(['Sum of ', num2str(a), ' and ', num2str(b), ' is ', num2str(a + b)]);

    % Create and display a matrix
    matrix = [1, 2, 3; 4, 5, 6; 7, 8, 9];
    disp('Matrix:');
    disp(matrix);

    % Compute and display eigenvalues
    eigenvalues = eig(matrix);
    disp('Eigenvalues of the matrix:');
    disp(eigenvalues);

    % Plot a sine wave
    x = 0:0.1:2*pi;  % Generate x values
    y = sin(x);      % Compute y values

    % figure;          % Open a new figure window
    % plot(x, y, 'LineWidth', 2);
    % title('Sine Wave');
    % xlabel('x');
    % ylabel('sin(x)');
    % grid on;
end