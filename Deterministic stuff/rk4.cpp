// #include <cmath>
// #include <functional>
#include <cmath>
#include <functional>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <valarray>
#include <vector>

#include <ginac/ginac.h>

template<typename T>
void rk4_step(
	std::function<T(const T, double)> f,
	T &x, double t, double dt
)
{
	T k1=f(x, t),
	  k2=f(x+0.5*dt*k1, t+0.5*dt),
	  k3=f(x+0.5*dt*k2, t+0.5*dt),
	  k4=f(x+dt*k3, t+dt);
	x+=dt/6*(k1+2*k2+2*k3+k4);
}

class Serializer
{
	std::vector<int> shape;
	int counter=0;

	public:

	Serializer(std::vector<int> shape):
		shape{shape}
	{}

	Serializer operator[](int i) const
	{
		std::vector<int> reduced_shape(shape.cbegin()+1, shape.cend());
		int step=std::accumulate(reduced_shape.cbegin(), reduced_shape.cend(), 1, std::multiplies<int>{});

		Serializer ret{reduced_shape};
		ret.counter = counter + step * i;

		return ret;
	}

	operator int() const
	{
		return counter;
	}

	int range() const
	{
		int step=std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>{});
		return counter+step;
	}
};

class StateSpace
{
	std::vector<GiNaC::realsymbol> coordinates,
	                               momentums;
	std::vector<std::string> cnames,
		                 mnames;

	void set_names(int num_degrees, std::vector<std::string> cnames, std::vector<std::string> mnames)
	{
		this->cnames=cnames;
		this->mnames=mnames;
		for(int i=cnames.size(); i<num_degrees; i++)
			cnames[i]=std::string("q")+std::to_string(i+1);
		for(int i=mnames.size(); i<num_degrees; i++)
			mnames[i]=std::string("p")+std::to_string(i+1);
	}

	void set_coordinates(int num_degrees)
	{
		for(int i=0; i<num_degrees; i++)
		{
			coordinates.push_back(GiNaC::realsymbol(cnames[i]));
			momentums.push_back(GiNaC::realsymbol(mnames[i]));
		}
	}

	public:
		StateSpace(int num_degrees=0, std::vector<std::string> cnames=std::vector<std::string>(), std::vector<std::string> mnames=std::vector<std::string>())
		{
			set_names(num_degrees, cnames, mnames);
			set_coordinates(num_degrees);
		}

		GiNaC::realsymbol &q(int i)
		{
			return coordinates[i];
		}

		GiNaC::realsymbol q(int i) const
		{
			return coordinates[i];
		}

		std::valarray<GiNaC::ex> valarray_q()
		{
			std::valarray<GiNaC::ex> coordinates(this->coordinates.size());
			for(int i=0; i<coordinates.size(); i++)
				coordinates[i] = this->coordinates[i];
			return coordinates;
		}

		GiNaC::realsymbol &p(int i)
		{
			return momentums[i];
		}

		GiNaC::realsymbol p(int i) const
		{
			return momentums[i];
		}

		std::valarray<GiNaC::ex> valarray_p()
		{
			std::valarray<GiNaC::ex> momentums(this->momentums.size());
			for(int i=0; i<momentums.size(); i++)
				momentums[i] = this->momentums[i];
			return momentums;
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

		std::string csv()
		{
			std::string ret;
			int dim=coordinates.size();
			for(int i=0; i<dim; i++)
				ret += cnames[i] + "," + mnames[i] + ",";
			return ret + t.get_name();
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
			values = std::valarray<double>(ss.dim() * 2);
		}

		double &q(int i)
		{
			return values[i];
		}

		double q(int i) const
		{
			return values[i];
		}

		double &p(int i)
		{
			int dim=values.size()/2;
			return values[dim+i];
		}

		double p(int i) const
		{
			int dim=values.size()/2;
			return values[dim+i];
		}

		int dim() const
		{
			return values.size()/2;
		}

		State operator+(const State &that) const
		{
			return State(this->values+that.values);
		}
		
		State &operator+=(const State &that)
		{
			values=values+that.values;
			return *this;
		}

		friend State operator*(double left, const State &right);

		friend std::ostream& operator<<(std::ostream &, const State &);

		std::string csv()
		{
			std::string ret;
			int dim=values.size()/2;
			for(int i=0; i<dim; i++)
				ret += std::to_string(values[i]) + "," + std::to_string(values[dim+i]) + ",";
			ret.resize(ret.size()-1);
			return ret;
		}
};

State operator*(double left, const State &right)
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

std::function<State(const State, double)> hamilton_equations(StateSpace &ss, GiNaC::ex hamiltonian)
{
	std::vector< std::function<double(const State, double)> > dq_dt(ss.dim()), dp_dt(ss.dim());
	std::vector<GiNaC::realsymbol> qs=ss.qs(), ps=ss.ps();
	GiNaC::realsymbol t=ss.t;
	for(int i=0; i<ss.dim(); i++)
	{
		GiNaC::ex dh_dq=hamiltonian.diff(ss.q(i)),
			dh_dp=hamiltonian.diff(ss.p(i));
		dq_dt[i]=[dh_dp, qs, ps, t](const State s, double t_)->double
			{
				GiNaC::lst subs;
				for(int i=0; i<qs.size(); i++)
				{
					subs.append(qs[i] == s.q(i));
					subs.append(ps[i] == s.p(i));
				}
				subs.append(t == t_);
				return GiNaC::ex_to<GiNaC::numeric>(dh_dp.subs(subs).evalf()).to_double();
			};
		dp_dt[i]=[dh_dq, qs, ps, t](const State s, double t_)->double
			{
				GiNaC::lst subs;
				for(int i=0; i<qs.size(); i++)
				{
					subs.append(qs[i] == s.q(i));
					subs.append(ps[i] == s.p(i));
				}
				subs.append(t == t_);
				return -GiNaC::ex_to<GiNaC::numeric>(dh_dq.subs(subs).evalf()).to_double();
			};
	}
	return [dq_dt, dp_dt](const State s, double t)->State
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

double radial_func(double dist)
{
	if(dist > 10) return 0;
	double sigma=1;
	return exp(-0.5*dist/sigma*sigma);
}

bool get_last_values(std::string fn, State &s, double &t)
{
	std::ifstream inf(fn);
	if(inf)
	{
		std::string last_line;
		std::string value;
		std::string line;
		while(std::getline(inf, line)) if(line != "") last_line=line;
		// F. it, we do it cavernicola style
		int dim=s.dim();
		int j=-1;
		for(int i=0; i<dim; i++)
		{
			value = "";
			for(j++;last_line[j]!=','; j++)
				value += last_line[j];
			s.q(i) = std::stod(value);
			value = "";
			for(j++;last_line[j]!=','; j++)
				value += last_line[j];
			s.p(i) = std::stod(value);
		}
		value = "";
		for(j++;last_line[j]!=','; j++)
			value += last_line[j];
		t = std::stod(value);
		return true;
	}
	return false;
}

int main()
{
	int width=20,
	    height=20;
	Serializer szr{{height, width}};
	int range=szr.range();
	std::vector<std::string> cnames(range), mnames(range);
	for(int i=0; i<height; i++) for(int j=0; j<width; j++)
	{
		cnames[szr[i][j]]=std::string("q_")+std::to_string(i)+"_"+std::to_string(j);
		mnames[szr[i][j]]=std::string("p_")+std::to_string(i)+"_"+std::to_string(j);
	}
	double v{1};
	StateSpace ss(range, cnames, mnames); // State space of dimension 1
	std::valarray<GiNaC::ex> q=ss.valarray_q(), p=ss.valarray_p();
	GiNaC::ex
		kinetic_energy{
			(p*p).sum()/(2*v*v)
		},
		potential_energy{
			0
		};

	for(int i=1; i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			GiNaC::ex neighbour_diff;
			neighbour_diff = q[szr[i-1][j]]-q[szr[i][j]];
			potential_energy += neighbour_diff * neighbour_diff;
			neighbour_diff = q[szr[j][i-1]]-q[szr[j][i]];
			potential_energy += neighbour_diff * neighbour_diff;
		}
		GiNaC::ex border_q;
		border_q = q[szr[i][0]];
		potential_energy += border_q*border_q;
		border_q = q[szr[i][width-1]];
		potential_energy += border_q*border_q;
		border_q=q[szr[0][i]];
		potential_energy += border_q*border_q;
		border_q=q[szr[width-1][i]];
		potential_energy += border_q*border_q;
	}
	potential_energy /= 2;

	GiNaC::ex hamiltonian{kinetic_energy + potential_energy};
	auto hamilton_eqs = hamilton_equations(ss, hamiltonian);
	State s(ss);
	for(int i=0; i<szr.range(); i++)
	{
		s.q(i)=0;
		s.p(i)=0;
	}
	for(int i=0; i<height; i++) for(int j=0; j<width; j++)
	{
		double mid_height=height%2?height/2:height/2-0.5,
		       mid_width=width%2?width/2.:width/2-0.5,
		       dist = pow(
			        (i-mid_height)*(i-mid_height)+
			        (j-mid_width)*(j-mid_width),
				0.5
			      );
		s.q(szr[i][j]) = radial_func(dist);
	}
	double t_offset = 0;
	double dt=0.05;

	std::ofstream outf;
	if(get_last_values("test", s, t_offset))
		outf.open("test", std::ios_base::app);
	else
	{
		outf.open("test");
		outf << ss.csv() << "\n";
	}
	for(int i=0; i<500; i++)
	{
		if(i%10==0) outf.flush();
		outf << s.csv() << "," << t_offset + i*dt << "\n";
		rk4_step(hamilton_eqs, s, t_offset + i*dt, dt);
	}
	outf.close();

	return 0;
}
