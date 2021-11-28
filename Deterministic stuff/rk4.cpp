// #include <cmath>
// #include <functional>
#include <functional>
#include <iostream>
#include <string>
#include <valarray>

#include <ginac/ginac.h>

template<typename T>
void rk4_step(
	std::function<T(const T&, double)> f,
	T &x, double t, double dt,
	std::function<T(const T&, const T&)> sum = [](const T &left, const T &right)->T{ return left+right; },
	std::function<T(double, const T&)> prod = [](double left, const T &right)->T{ return left*right; }
)
{
	T k1=f(x, t),
	  k2=f(sum(x, prod(0.5*dt, k1)), t+0.5*dt),
	  k3=f(sum(x, prod(0.5*dt, k2)), t+0.5*dt),
	  k4=f(sum(x, prod(dt, k3)), t+dt);
	x = sum(x, prod(1./6, sum(sum(k1, prod(2, k2)), sum(prod(2, k3), k4))));
	x+=1./6*(k1+2*k2+2*k3+k4);
}

class StateSpace
{
	std::vector<GiNaC::realsymbol> coordinates;
	std::vector<GiNaC::realsymbol> momentums;

	public:
		StateSpace(int num_degrees=0)
		{
			for(int i=0; i<num_degrees; i++)
			{
				coordinates.push_back(GiNaC::realsymbol(std::string("q")+std::to_string(i+1)));
				momentums.push_back(GiNaC::realsymbol(std::string("p")+std::to_string(i+1)));
			}
		}

		GiNaC::realsymbol q(int i)
		{
			return coordinates[i];
		}

		GiNaC::realsymbol p(int i)
		{
			return momentums[i];
		}

		GiNaC::realsymbol t("t");

		std::vector<GiNaC::realsymbol> qs()
		{
			return coordinates;
		}

		std::vector<GiNaC::realsymbol> ps()
		{
			return momentums;
		}

		int dim()
		{
			return coordinates.size();
		}

		int add_dim()
		{
			int dim=coordinates.size();
			coordinates.push_back(GiNaC::realsymbol(std::string("q")+std::to_string(dim)));
			momentums.push_back(GiNaC::realsymbol(std::string("p")+std::to_string(dim)));
			return dim+1;
		}

		int add_dims(int dims=1)
		{
			int dim=coordinates.size();
			for(int i=dim; i<dim+dims; i++)
			{
				coordinates.push_back(GiNaC::realsymbol(std::string("q")+std::to_string(i)));
				momentums.push_back(GiNaC::realsymbol(std::string("p")+std::to_string(i)));
			}
			return dim+dims;
		}
};

class State
{
	std::valarray<double> values;

	public:
		double t;

		State(int num_degrees=0)
		{
			values = std::valarray<double>(num_degrees);
		}

		State(std::valarray<double> values)
		{
			if(values.size()%2)
				throw std::logic_error("Trying to force-initialize ill-formed State");
			this->values=values;
		}

		State(StateSpace ss)
		{
			values = std::valarray<double>(ss.dim());
		}

		double get_q(int i)
		{
			return values[i];
		}

		double get_p(int i)
		{
			int dim=values.size()/2;
			return values[dim+i];
		}

		void set_q(int i, double val)
		{
			values[i]=val;
		}

		void set_p(int i, double val)
		{
			int dim=values.size()/2;
			values[dim+i]=val;
		}

		int dim()
		{
			return values.size()/2;
		}

		State operator+(const State &that)
		{
			return State(this->values+that.values);
		}

		friend State operator*(const double left, const State &right);
};

State operator*(const double left, const State &right)
{
	return State(left*right.values);
}

std::function<State(const State&, double)> hamilton_equations(const StateSpace &ss, GiNaC::ex hamiltonian)
{
	std::vector< std::function<double(const State&, double)> > dq_dt(ss.dim()), dp_dt(ss.dim());
	std::vector<GiNaC::realsymbol> qs=ss.qs(), ps=ss.ps();
	GiNaC::realsymbol t=ss.t;
	for(int i=0; i<ss.dim(); i++)
	{
		GiNaC::ex dh_dq=hamiltonian.diff(ss.q(i)),
			dh_dp=hamiltonian.diff(ss.p(i));
		dq_dt.push_back([dh_dp, qs, ps, t](const State &s, double t_)->double
			{
				GiNaC::lst subs;
				for(int i=0; i<qs.size(); i++)
				{
					subs.append(qs[i] == s.get_q(i));
					subs.append(ps[i] == s.get_p(i));
				}
				subs.append(t == t_);
				return GiNaC::evalf(dh_dp.subs(subs));
			}
		);
		dp_dt.push_back([dh_dq, qs, ps, t](const State &s, double t_)->double
			{
				GiNaC::lst subs;
				for(int i=0; i<qs.size(); i++)
				{
					subs.append(qs[i] == s.get_q(i));
					subs.append(ps[i] == s.get_p(i));
				}
				subs.append(t == t_);
				return -GiNaC::evalf(dh_dp.subs(subs));
			}
		);
	}
	return [dq_dt, dp_dt](const State &s, double t)->State
	{
		int dim=s.dim();
		State ret(dim);
		for(int i=0; i<dim; i++)
		{
			ret.set_q(i, dq_dt(s, t));
			ret.set_p(i, dp_dt(s, t));
		}
		return ret;
	}
}

/*
 * Quiero que las partículas partan con un movimiento rotacional alrededor de cada neurona.
 * La distancia definida por un potencial central tipo -α/r+β/r^2.
 * Para estabilizar, quitarles energía manualmente.
 * Las neuronas conectadas por o bien una recta, o la trayectoria de una partícula que conectó con movimiento browniano forzado ambos caminos.
 * Ese conector ahora me ayudaría a definir un potencial que efectivamente lleva a las partículas desde una neurona hacia la otra.
 * Esto hacerlo quizás de a pares, o en un escenario global. Preferiblemente tridimensional :drool:
 * Quizas meterles movimiento browniano a las moscas también.
 */

int main()
{
	using namespace GiNaC;
	auto ss = StateSpace(1);

	std::cout << ss.q(0) << std::endl;

	realsymbol x, y;
	ex poly;

	for(int i=0; i<3; i++)
		poly += factorial(i+16)*pow(x, i)*pow(y, 2-i);

	// std::cout << poly << std::endl;
	// std::cout << poly.subs(x == 1).subs(y == 3) << std::endl;
	return 0;
}
