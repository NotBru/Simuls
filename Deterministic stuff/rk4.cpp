// #include <cmath>
// #include <functional>
#include <functional>
#include <iostream>
#include <string>
#include <valarray>

#include <ginac/ginac.h>

template<typename T>
void rk4_step(
	std::function<T(T&, double)> f,
	T &x, double t, double dt,
	std::function<T(T&, T&)> sum = [](T &left, T &right)->T{ return left+right; },
	std::function<T(double, T&)> prod = [](double left, T &right)->T{ return left*right; }
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

		GiNaC::realsymbol &q(int i)
		{
			return coordinates[i];
		}

		GiNaC::realsymbol &p(int i)
		{
			return momentums[i];
		}

		GiNaC::realsymbol t=GiNaC::realsymbol("t");

		std::vector<GiNaC::realsymbol> qs()
		{
			return coordinates;
		}

		std::vector<GiNaC::realsymbol> ps()
		{
			return momentums;
		}

		int dim() const
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

	State(std::valarray<double> values)
	{
		if(values.size()%2)
			throw std::logic_error("Trying to force-initialize ill-formed State");
		this->values=values;
	}

	public:
		double t;

		State(int num_degrees=0)
		{
			values = std::valarray<double>(num_degrees * 2);
		}

		State(StateSpace ss)
		{
			values = std::valarray<double>(ss.dim());
		}

		double &q(int i)
		{
			return values[i];
		}

		double &p(int i)
		{
			int dim=values.size()/2;
			return values[dim+i];
		}

		int dim() const
		{
			return values.size()/2;
		}

		State operator+(State &that)
		{
			return State(this->values+that.values);
		}

		friend State operator*(double left, State &right);

		friend std::ostream& operator<<(std::ostream &, const State &);
};

State operator*(double left, State &right)
{
	return State(left*right.values);
}

std::ostream& operator<<(std::ostream &out, const State &s)
{
	int dim=s.dim();
	out << "[";
	if(dim>0)
		out << "(" << s.values[0] << ", " << s.values[dim] << ")";
	for(int i=1; i<dim; i++)
		out << ", (" << s.values[i] << ", " << s.values[dim+i] << ")";
	out << "]";
	return out;
}

std::function<State(State&, double)> hamilton_equations(StateSpace &ss, GiNaC::ex hamiltonian)
{
	std::vector< std::function<double(State&, double)> > dq_dt(ss.dim()), dp_dt(ss.dim());
	std::vector<GiNaC::realsymbol> qs=ss.qs(), ps=ss.ps();
	GiNaC::realsymbol t=ss.t;
	for(int i=0; i<ss.dim(); i++)
	{
		GiNaC::ex dh_dq=hamiltonian.diff(ss.q(i)),
			dh_dp=hamiltonian.diff(ss.p(i));
		dq_dt.push_back([dh_dp, qs, ps, t](State &s, double t_)->double
			{
				GiNaC::lst subs;
				for(int i=0; i<qs.size(); i++)
				{
					subs.append(qs[i] == s.q(i));
					subs.append(ps[i] == s.p(i));
				}
				subs.append(t == t_);
				return GiNaC::ex_to<GiNaC::numeric>(dh_dp.subs(subs).evalf()).to_double();
			}
		);
		dp_dt.push_back([dh_dq, qs, ps, t](State &s, double t_)->double
			{
				GiNaC::lst subs;
				for(int i=0; i<qs.size(); i++)
				{
					subs.append(qs[i] == s.q(i));
					subs.append(ps[i] == s.p(i));
				}
				subs.append(t == t_);
				return -GiNaC::ex_to<GiNaC::numeric>(dh_dq.subs(subs).evalf()).to_double();
			}
		);
	}
	return [dq_dt, dp_dt](State &s, double t)->State
	{
		int dim=s.dim();
		State ret(dim);
		for(int i=0; i<dim; i++)
		{
			ret.q(i)=dq_dt[i](s, t);
			ret.p(i)=dp_dt[i](s, t);
		}
		return ret;
	};
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
	auto hamiltonian = .5 * ss.p(0) * ss.p(0) + .5 * ss.q(0) * ss.q(0);
	auto hamilton_eqs = hamilton_equations(ss, hamiltonian);

	std::cout << hamiltonian << std::endl;
	return 0;
}
