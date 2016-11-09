#include"common.h"


double random(double start, double end)
{
	return start + (end-start)*rand()/(RAND_MAX + 1);
}
