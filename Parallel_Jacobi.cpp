#include <omp.h>
#include <cmath>
#include <stdio.h>
#include <time.h>

#define ITERATION_LIMIT 333
#define EPSILON 0.0000001
#define NUM_OF_THREADS 4

bool check_diagoanally_dominant_sequential(float** matrix, int matrix_size);
bool check_diagoanally_dominant_parallel(float** matrix, int matrix_size);
void solve_jacobi_sequential(float** matrix, int matrix_size, float* right_hand_side);
void solve_jacobi_parallel(float** matrix, int matrix_size, float* right_hand_side);
void init_array_sequential(float array[], int array_size);
float* clone_array_sequential(float array[], int array_length);
void init_array_parallel(float array[], int array_size);
float* clone_array_parallel(float array[], int array_length);
void delete_matrix(float** matrix, int matrix_size);

int main(){
	// Just to simulate a real program
	int user_choice = 1;
	while (user_choice == 1)
	{
		int matrix_size;
		// Entering the matrix size .. 
		printf("Enter the matrix size: ");
		scanf_s("%d", &matrix_size);
		
		// Initializing the main structures .. 
		float** matrix = new float*[matrix_size];
		for (int i = 0; i < matrix_size; i++) matrix[i] = new float[matrix_size];
		float* right_hand_side = new float[matrix_size];

		// Entering the matrix elements ..
		printf("Enter the matrix elements (Enter element by element and row by row from the matrix top left corner to its end by pressing ENTER):\n");
		for (int i = 0; i < matrix_size; i++){
			printf("Row #%d elements:------\n", i);
			for (int j = 0; j < matrix_size; j++) {
				printf("Matrix[%d][%d]:\n", i, j);
				scanf_s("%f", &matrix[i][j]);
			}
		}

		// Printing the matrix (As a confirmation for the user) .. 
		printf("The matrix in its final shape: \n");
		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				printf("%f ", matrix[i][j]);
			}
			printf("\n");
		}

		// Checking if the matrix is diagonally dominant or not ..
		// I prefered to use the parallel dominant check version here .. 
		// It won't affact the comparison between the 2 modes as we are using the same function in the 2 modes.
		omp_set_num_threads(NUM_OF_THREADS);
		if (!check_diagoanally_dominant_parallel(matrix, matrix_size)){
			printf("The matrix is not dominant, please enter another matrix.\n");
			// We don't need the matrix if it's not diagonally dominant .. 
			// Cleaning the chaos
			delete_matrix(matrix, matrix_size);
			delete[] right_hand_side;
			printf("Do you want to continue? 1/0\n");
			scanf_s("%d", &user_choice);
			continue;
		}

		// Allowing the user to enter the RHS .. 
		printf("Enter the right hand side (it has a length of %d):\n", matrix_size);
		for (int i = 0; i < matrix_size; i++){
			printf("Element #%d\n", i);
			scanf_s("%f", &right_hand_side[i]);
		}

		// Entering the running mode .. 
		printf("Choose: Serial Mode -> 0, Parallel Mode -> 1 :\nYour Choice: \n");
		int run_mode_choice;
		scanf_s("%d", &run_mode_choice);

		switch (run_mode_choice)
		{
			// Serial run..
			case 0:
			{
				// Computing the time
				const clock_t serial_starting_time = clock();
				solve_jacobi_sequential(matrix, matrix_size, right_hand_side);
				printf("Elapsed time: %f ms\n", float(clock() - serial_starting_time));
			}
			break;
			// Parallel run
			case 1:
			{
				// Computing the time
				const clock_t parallel_starting_time = clock();
				// Initializing the parallel mode in case of it was chosen ..
				omp_set_num_threads(NUM_OF_THREADS);
				solve_jacobi_parallel(matrix, matrix_size, right_hand_side);
				printf("Elapsed time: %f ms\n", float(clock() - parallel_starting_time));
			}
			break;
		}

		// Cleaning the chaos
		delete_matrix(matrix, matrix_size);
		delete[] right_hand_side;

		printf("Do you want to continue? 1/0\n");
		scanf_s("%d", &user_choice);
	}
}

bool check_diagoanally_dominant_parallel(float** matrix, int matrix_size){
	// This is to validate that all the rows applies the rule .. 
	int check_count = 0;
	#pragma omp parallel 
	{
		// For each row
		// Each thread will be assigned to run on a row.
		#pragma omp for schedule (guided, 1)
		for (int i = 0; i < matrix_size; i++){
			float row_sum = 0;
			// Summing the other row elements .. 
			for (int j = 0; j < matrix_size; j++) {
				if (j != i) row_sum += abs(matrix[i][j]);
			}

			if (abs(matrix[i][i]) >= row_sum){
				#pragma omp atomic 
				check_count++;
			}
		}
	}
	return check_count == matrix_size;
}

bool check_diagoanally_dominant_sequential(float** matrix, int matrix_size){
	int check_count = 0;
	// For each row ..
	for (int i = 0; i < matrix_size; i++) {
		float row_sum = 0;
		// Summing the other row elements .. 
		for (int j = 0; j < matrix_size; j++) {
			if (j != i) row_sum += abs(matrix[i][j]);
		}

		if (abs(matrix[i][i]) >= row_sum) {
			check_count++;
		}
	}
	return check_count == matrix_size;
}

void solve_jacobi_sequential(float** matrix, int matrix_size, float* right_hand_side) {
	float* solution = new float[matrix_size];
	float* last_iteration = new float[matrix_size];
	
	// Just for initialization ..
	printf("Iterations:----------------------------------------- \n");
	init_array_sequential(solution, matrix_size);
	
	for (int i = 0; i < ITERATION_LIMIT; i++){
		last_iteration = clone_array_sequential(solution, matrix_size);
		for (int j = 0; j < matrix_size; j++) {
			float sigma_value = 0;
			for (int k = 0; k < matrix_size; k++) {
				if (j != k) {
					sigma_value += matrix[j][k] * solution[k];
				}
			}
			solution[j] = (right_hand_side[j] - sigma_value) / matrix[j][j];
		}

		// Checking for the stopping condition ...
		int stopping_count = 0;
		for (int s = 0; s < matrix_size; s++) {
			if (abs(last_iteration[s] - solution[s]) <= EPSILON) {
				stopping_count++;
			}
		}

		if (stopping_count == matrix_size) break;

		printf("Iteration #%d: ", i+1);
		for (int l = 0; l < matrix_size; l++) {
			printf("%f ", solution[l]);
		}
		printf("\n");
	}
}

void solve_jacobi_parallel(float** matrix, int matrix_size, float* right_hand_side) {
	float* solution = new float[matrix_size];
	float* last_iteration = new float[matrix_size];

	// Just for initialization ..
	printf("Iterations:--------------------------------------------------\n");
	init_array_parallel(solution, matrix_size); // dump the array with zeroes

	// NOTE: we don't need to parallelize this as the iterations are dependent. However, we may parallelize the inner processes 
	for (int i = 0; i < ITERATION_LIMIT; i++){
		// Make a deep copy to a temp array to compare it with the resulted vector later
		last_iteration = clone_array_parallel(solution, matrix_size);
		
		// Each thread is assigned to a row to compute the corresponding solution element
		#pragma omp parallel for schedule(dynamic, 1)
		for (int j = 0; j < matrix_size; j++){
			float sigma_value = 0;
			for (int k = 0; k < matrix_size; k++){
				if (j != k) {
					sigma_value += matrix[j][k] * solution[k];
				}
			}
			solution[j] = (right_hand_side[j] - sigma_value) / matrix[j][j];
		}

		// Checking for the stopping condition ...
		int stopping_count = 0;
		#pragma omp parallel for schedule(dynamic, 1) 
		for (int s = 0; s < matrix_size; s++) {
			if (abs(last_iteration[s] - solution[s]) <= EPSILON) {
				#pragma atomic
				stopping_count++;
			}
		}

		if (stopping_count == matrix_size) break;

		printf("Iteration #%d: ", i+1);
		for (int l = 0; l < matrix_size; l++) {
			printf("%f ", solution[l]);
		}
		printf("\n");
	}
}

void init_array_sequential(float array[], int array_size){
	for (int i = 0; i < array_size; i++) {
		array[i] = 0;
	}
}

float* clone_array_sequential(float array[], int array_length){
	float* output = new float[array_length];
	for (int i = 0; i < array_length; i++) {
		output[i] = array[i];
	}
	return output;
}

void init_array_parallel(float array[], int array_size){
	#pragma omp parallel for schedule (dynamic, 1)
	for (int i = 0; i < array_size; i++) {
		array[i] = 0;
	}
}

float* clone_array_parallel(float array[], int array_length){
	float* output = new float[array_length];
	#pragma omp parallel for schedule (dynamic, 1)
	for (int i = 0; i < array_length; i++) {
		output[i] = array[i];
	}
	return output;
}

void delete_matrix(float** matrix, int matrix_size) {
	for (int i = 0; i < matrix_size; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}