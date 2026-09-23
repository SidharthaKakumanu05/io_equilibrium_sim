/* Exact-operation IO integration. No fast math or contracted multiply-add.
 * NumPy's public exp ufunc and matmul keep CPU dispatch and reduction behavior.
 * Python io_channels.IOPopulation.substep is the executable reference.
 */
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
static PyObject *exp_func;
static double scalar(PyObject *o,const char *key){PyObject *v=PyObject_GetAttrString(o,key); if(!v)return 0.;double x=PyFloat_AsDouble(v);Py_DECREF(v);return x;}
static PyArrayObject *arr(PyObject *o,const char *key){return (PyArrayObject*)PyObject_GetAttrString(o,key);}
static double at(PyObject *o,npy_intp i){
 if(PyArray_Check(o)){
  PyArrayObject *a=(PyArrayObject*)o;
  return *(double*)(PyArray_BYTES(a)+(PyArray_SIZE(a)==1?0:i*PyArray_STRIDE(a,0)));
 }
 return PyFloat_AsDouble(o);
}
static int vector_ok(PyObject *o,npy_intp n,int mutable){
 return o && PyArray_Check(o) && PyArray_TYPE((PyArrayObject*)o)==NPY_DOUBLE
  && PyArray_NDIM((PyArrayObject*)o)==1 && PyArray_SIZE((PyArrayObject*)o)==n
  && (!mutable || (PyArray_IS_C_CONTIGUOUS((PyArrayObject*)o) && PyArray_ISWRITEABLE((PyArrayObject*)o)));
}
static int broadcast_ok(PyObject *o,npy_intp n){
 if(!o)return 0;
 if(!PyArray_Check(o))return PyFloat_Check(o)||PyLong_Check(o);
 PyArrayObject *a=(PyArrayObject*)o;
 return PyArray_TYPE(a)==NPY_DOUBLE && PyArray_NDIM(a)<=1 && (PyArray_SIZE(a)==1 || PyArray_SIZE(a)==n);
}
static int exponential(PyArrayObject *input,PyArrayObject *output){PyObject*r=PyObject_CallFunctionObjArgs(exp_func,input,output,NULL);if(!r)return -1;Py_DECREF(r);return 0;}
static PyObject *advance(PyObject *module,PyObject *args){
 PyObject *pop,*gaba_obj,*noise_obj;int substeps;
 if(!PyArg_ParseTuple(args,"OOOi",&pop,&gaba_obj,&noise_obj,&substeps))return NULL;
 PyObject *p=PyObject_GetAttrString(pop,"_p"),*gap=PyObject_GetAttrString(pop,"g_gap"),*rows=PyObject_GetAttrString(pop,"_gap_row_sum");
 PyObject *gc=PyObject_GetAttrString(pop,"_g_cal"),*gh=PyObject_GetAttrString(pop,"_g_cah"),*gk=PyObject_GetAttrString(pop,"_g_kca"),*gq=PyObject_GetAttrString(pop,"_g_h");
 PyArrayObject *oldv=arr(pop,"V"),*oldca=arr(pop,"Ca"),*oldnoise=arr(pop,"noise");
 PyArrayObject *k=arr(pop,"k"),*l=arr(pop,"l"),*r=arr(pop,"r"),*s=arr(pop,"s"),*q=arr(pop,"q"),*ref=arr(pop,"_refractory_ms");
 PyArrayObject *v=NULL,*ca=NULL,*noise=NULL,*fired=NULL,*in=NULL,*out=NULL;
 double *scratch=NULL;int failed=1;
 if(!oldv || !PyArray_Check(oldv) || PyArray_NDIM(oldv)!=1){
  if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"IO V must be a float64 vector");goto cleanup;
 }
 npy_intp n=PyArray_SIZE(oldv),dims[1]={n};
 PyObject *states[]={(PyObject*)oldv,(PyObject*)oldca,(PyObject*)oldnoise,(PyObject*)k,(PyObject*)l,(PyObject*)r,(PyObject*)s,(PyObject*)q,(PyObject*)ref};
 for(int j=0;j<9;j++)if(!vector_ok(states[j],n,j>=3)){
  if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"Native IO requires float64 population vectors and contiguous writable gates");goto cleanup;
 }
 PyObject *broadcasts[]={gc,gh,gk,gq,rows,gaba_obj};
 for(int j=0;j<6;j++)if(!broadcast_ok(broadcasts[j],n)){
  if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"Invalid native IO conductance array");goto cleanup;
 }
 if(!p || !gap || !PyArray_Check(noise_obj) || PyArray_TYPE((PyArrayObject*)noise_obj)!=NPY_DOUBLE || PyArray_NDIM((PyArrayObject*)noise_obj)!=2 || PyArray_DIM((PyArrayObject*)noise_obj,0)!=substeps || PyArray_DIM((PyArrayObject*)noise_obj,1)!=n){
  if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"Invalid native IO noise block");goto cleanup;
 }
 v=(PyArrayObject*)PyArray_NewCopy(oldv,NPY_CORDER);ca=(PyArrayObject*)PyArray_NewCopy(oldca,NPY_CORDER);noise=(PyArrayObject*)PyArray_NewCopy(oldnoise,NPY_CORDER);
 fired=(PyArrayObject*)PyArray_ZEROS(1,dims,NPY_BOOL,0);in=(PyArrayObject*)PyArray_SimpleNew(1,dims,NPY_DOUBLE);out=(PyArrayObject*)PyArray_SimpleNew(1,dims,NPY_DOUBLE);
 if(!v||!ca||!noise||!fired||!in||!out)goto cleanup;
 double *V=PyArray_DATA(v),*Ca=PyArray_DATA(ca),*N=PyArray_DATA(noise),*K=PyArray_DATA(k),*L=PyArray_DATA(l),*R=PyArray_DATA(r),*S=PyArray_DATA(s),*Q=PyArray_DATA(q),*Ref=PyArray_DATA(ref),*X=PyArray_DATA(in),*E=PyArray_DATA(out);npy_bool*F=PyArray_DATA(fired);
 scratch=PyMem_Malloc(sizeof(double)*(n?n:1)*9);if(!scratch){PyErr_NoMemory();goto cleanup;}
 double *vg=scratch,*gt=vg+n,*vinf=gt+n,*vn=vinf+n,*gcah=vn+n,*a=gcah+n,*b=a+n,*c=b+n,*ra=c+n;
 double dt=scalar(pop,"sub_dt_ms"),nd=scalar(pop,"_noise_decay"),nk=scalar(pop,"_noise_kick"),ks=scalar(pop,"_k_step"),cd=scalar(pop,"_ca_decay_factor");
 double gleak=scalar(p,"g_leak"),eca=scalar(p,"e_ca"),ek=scalar(p,"e_k"),eh=scalar(p,"e_h"),el=scalar(p,"e_leak"),eg=scalar(p,"e_gaba"),ia=scalar(p,"i_app"),cm=scalar(p,"c_m"),ci=scalar(p,"ca_influx"),decay=scalar(p,"ca_decay"),cs=scalar(p,"kca_ca_scale"),am=scalar(p,"kca_alpha_max"),sb=scalar(p,"kca_beta"),vs=scalar(p,"v_spike_mv"),sr=scalar(p,"spike_refractory_ms");
 if(PyErr_Occurred())goto cleanup;
 failed=0;
 for(int step=0;step<substeps;step++){
  PyArrayObject *gapv=NULL;
  if(gap!=Py_None){gapv=(PyArrayObject*)PyNumber_MatrixMultiply(gap,(PyObject*)v);if(!gapv){failed=1;break;}}
  for(npy_intp i=0;i<n;i++){
   double vv=V[i];vg[i]=vv < -200. ? -200. : vv > 100. ? 100. : vv;
   N[i]=N[i]*nd+nk*(*(double*)PyArray_GETPTR2((PyArrayObject*)noise_obj,step,i));
   double cal=at(gc,i)*K[i]*K[i]*K[i]*L[i],cah=at(gh,i)*R[i]*R[i],kca=at(gk,i)*S[i],hh=at(gq,i)*Q[i],gg=at(gaba_obj,i);
   gcah[i]=cah;gt[i]=cal+cah+kca+hh+gleak+gg+at(rows,i);
   double ii=cal*eca+cah*eca+kca*ek+hh*eh+gleak*el+gg*eg+ia+N[i];
   if(gapv)ii=ii+((double*)PyArray_DATA(gapv))[i];vinf[i]=ii/gt[i];X[i]=-dt*gt[i]/cm;
  }
  Py_XDECREF(gapv);
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){vn[i]=vinf[i]+(V[i]-vinf[i])*E[i]; double ic=gcah[i]*(V[i]-eca); double cif=-ci*ic/decay; Ca[i]=cif+(Ca[i]-cif)*cd; if(Ca[i]<=0.)Ca[i]=0.;}
  for(npy_intp i=0;i<n;i++){X[i]=-(vg[i]+61.0)/4.2;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){K[i]+=(1.0/(1.0+E[i])-K[i])*ks;}
  for(npy_intp i=0;i<n;i++){X[i]=(vg[i]+85.5)/8.5;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){a[i]=1.0/(1.0+E[i]);}
  for(npy_intp i=0;i<n;i++){X[i]=(vg[i]+160.0)/30.0;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){b[i]=20.0*E[i];}
  for(npy_intp i=0;i<n;i++){X[i]=(vg[i]+84.0)/7.3;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){c[i]=b[i]/(1.0+E[i])+35.0;}
  for(npy_intp i=0;i<n;i++){X[i]=-dt/c[i];}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){L[i]+=(a[i]-L[i])*(1.0-E[i]);}
  for(npy_intp i=0;i<n;i++){X[i]=-(vg[i]-5.0)/13.9;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){ra[i]=1.7/(1.0+E[i]);}
  for(npy_intp i=0;i<n;i++){X[i]=(vg[i]+8.5)/5.0;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){double xx=vg[i]+8.5;double rb=fabs(xx)<1e-6?0.1:0.02*xx/(E[i]-1.0);b[i]=ra[i]+rb;}
  for(npy_intp i=0;i<n;i++){X[i]=-dt*b[i];}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){R[i]+=(ra[i]/b[i]-R[i])*(1.0-E[i]);}
  for(npy_intp i=0;i<n;i++){X[i]=(vg[i]+80.0)/4.0;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){a[i]=1.0/(1.0+E[i]);}
  for(npy_intp i=0;i<n;i++){X[i]=-0.086*vg[i]-14.6;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){b[i]=E[i];}
  for(npy_intp i=0;i<n;i++){X[i]=0.070*vg[i]-1.87;}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){c[i]=1.0/(b[i]+E[i]);}
  for(npy_intp i=0;i<n;i++){X[i]=-dt/c[i];}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){Q[i]+=(a[i]-Q[i])*(1.0-E[i]);}
  for(npy_intp i=0;i<n;i++){a[i]=cs*Ca[i];if(a[i]>am)a[i]=am;b[i]=a[i]+sb;}
  for(npy_intp i=0;i<n;i++){X[i]=-dt*b[i];}
  if(exponential(in,out)<0){failed=1;break;}
  for(npy_intp i=0;i<n;i++){S[i]+=(a[i]/b[i]-S[i])*(1.0-E[i]); Ref[i]=Ref[i]-dt;if(Ref[i]<=0.)Ref[i]=0.; if(V[i]<vs && vn[i]>=vs && Ref[i]<=0.){F[i]=1;Ref[i]=sr;}V[i]=vn[i];}

 }
 if(!failed){if(PyObject_SetAttrString(pop,"V",(PyObject*)v)<0 || PyObject_SetAttrString(pop,"Ca",(PyObject*)ca)<0 || PyObject_SetAttrString(pop,"noise",(PyObject*)noise)<0)failed=1;}
 cleanup:
 PyMem_Free(scratch);Py_XDECREF(p);Py_XDECREF(gap);Py_XDECREF(rows);Py_XDECREF(gc);Py_XDECREF(gh);Py_XDECREF(gk);Py_XDECREF(gq);Py_XDECREF(oldv);Py_XDECREF(oldca);Py_XDECREF(oldnoise);Py_XDECREF(k);Py_XDECREF(l);Py_XDECREF(r);Py_XDECREF(s);Py_XDECREF(q);Py_XDECREF(ref);Py_XDECREF(v);Py_XDECREF(ca);Py_XDECREF(noise);Py_XDECREF(in);Py_XDECREF(out);
 if(failed || PyErr_Occurred()){Py_XDECREF(fired);return NULL;}return (PyObject*)fired;
}
/* PCG XSL RR 128/64 recurrence, following NumPy 2.2.6 pcg64.h.
 * Public state integers only: no dependency on NumPy's private struct layout.
 * Four interleaved subsequences remove the serial dependency without changing
 * draw order. Every output is converted exactly like NumPy's uint64_to_double.
 */
static PyObject *bernoulli(PyObject *module,PyObject *args){
 PyObject *shape;double probability;unsigned long long sh,sl,ih,il;PyArray_Dims dims={NULL,0};
 if(!PyArg_ParseTuple(args,"OdKKKK",&shape,&probability,&sh,&sl,&ih,&il))return NULL;
 if(!PyArray_IntpConverter(shape,&dims))return NULL;
 PyArrayObject *out=(PyArrayObject*)PyArray_SimpleNew(dims.len,dims.ptr,NPY_BOOL);PyDimMem_FREE(dims.ptr);if(!out)return NULL;
 npy_intp size=PyArray_SIZE(out);npy_bool *data=PyArray_DATA(out);
 __uint128_t state=((__uint128_t)sh<<64)|sl,inc=((__uint128_t)ih<<64)|il;
 const __uint128_t mul=((__uint128_t)2549297995355413924ULL<<64)|4865540595714422341ULL;
 __uint128_t m2=mul*mul,m4=m2*m2,inc4=inc*(1+mul+m2+m2*mul);
 __uint128_t lanes[4];lanes[0]=state*mul+inc;
 for(int j=1;j<4;j++)lanes[j]=lanes[j-1]*mul+inc;
 npy_intp i=0;
 Py_BEGIN_ALLOW_THREADS
 for(;i+3<size;i+=4){
  for(int j=0;j<4;j++){
   uint64_t x=(uint64_t)(lanes[j]>>64)^(uint64_t)lanes[j];unsigned rot=lanes[j]>>122;
   uint64_t raw=(x>>rot)|(x<<((-rot)&63));
   data[i+j]=(raw>>11)*0x1.0p-53<probability;
  }
  state=lanes[3];
  for(int j=0;j<4;j++)lanes[j]=lanes[j]*m4+inc4;
 }
 for(;i<size;i++){
  state=state*mul+inc;uint64_t x=(uint64_t)(state>>64)^(uint64_t)state;unsigned rot=state>>122;
  uint64_t raw=(x>>rot)|(x<<((-rot)&63));data[i]=(raw>>11)*0x1.0p-53<probability;
 }
 Py_END_ALLOW_THREADS
 return Py_BuildValue("NKK",out,(unsigned long long)(state>>64),(unsigned long long)state);
}
/* Eligibility and RNG draws stay in Python. This consumes the already-resolved
 * unique indices and uniforms in their original order, including endpoints.
 */
static PyObject *discrete(PyObject *module,PyObject *args){
 PyArrayObject *weights,*states,*counts,*ids,*directions,*uniforms;int mode;
 double pd,pu,low,high;
 if(!PyArg_ParseTuple(args,"O!O!O!O!O!O!idddd",&PyArray_Type,&weights,&PyArray_Type,&states,&PyArray_Type,&counts,&PyArray_Type,&ids,&PyArray_Type,&directions,&PyArray_Type,&uniforms,&mode,&pd,&pu,&low,&high))return NULL;
 npy_intp n=PyArray_SIZE(ids);
 if(PyArray_NDIM(weights)!=2 || PyArray_TYPE(weights)!=NPY_DOUBLE || !PyArray_ISWRITEABLE(weights)
  || PyArray_NDIM(states)!=2 || PyArray_TYPE(states)!=NPY_UINT8 || !PyArray_ISWRITEABLE(states)
  || PyArray_NDIM(counts)!=2 || PyArray_TYPE(counts)!=NPY_INT64 || !PyArray_ISWRITEABLE(counts)
  || PyArray_NDIM(ids)!=1 || PyArray_TYPE(ids)!=NPY_INTP
  || PyArray_NDIM(directions)!=1 || PyArray_TYPE(directions)!=NPY_INT8 || PyArray_SIZE(directions)!=n
  || PyArray_NDIM(uniforms)!=1 || PyArray_TYPE(uniforms)!=NPY_DOUBLE || PyArray_SIZE(uniforms)!=n
  || mode<0 || mode>2 || PyArray_DIM(weights,0)!=PyArray_DIM(states,0) || PyArray_DIM(weights,1)!=PyArray_DIM(states,1)
  || PyArray_DIM(weights,0)!=PyArray_DIM(counts,0) || PyArray_DIM(weights,1)!=PyArray_DIM(counts,1)){
  PyErr_SetString(PyExc_ValueError,"Invalid native discrete-update arrays");return NULL;
 }
 const unsigned char down[2][8]={{0,0,1,2,3,3,3,3},{0,0,1,2,3,4,5,6}};
 const unsigned char up[2][8]={{4,4,4,4,5,6,7,7},{1,2,3,4,5,6,7,7}};
 const double md[8]={0,.125,.25,.5,1,.5,.25,.125},mu[8]={.125,.25,.5,1,.5,.25,.125,0};
 npy_intp width=PyArray_DIM(weights,1),size=PyArray_SIZE(weights);
 long long transitions=0,switches=0,down_switches=0,up_switches=0;
 for(npy_intp i=0;i<n;i++){
  npy_intp id=*(npy_intp*)PyArray_GETPTR1(ids,i);
  if(id<0 || id>=size){PyErr_SetString(PyExc_ValueError,"Invalid synapse index");return NULL;}
  npy_intp row=id/width,col=id%width;
  unsigned char *state=PyArray_GETPTR2(states,row,col),before=*state;
  if(before>7){PyErr_SetString(PyExc_ValueError,"Invalid cascade state");return NULL;}
  int ltd=(*(npy_int8*)PyArray_GETPTR1(directions,i))==-1;
  double probability=ltd?pd:pu;
  unsigned char target=ltd?3:4;
  if(mode){probability=(ltd?md[before]:mu[before])*probability;target=ltd?down[mode-1][before]:up[mode-1][before];}
  double uniform=*(double*)PyArray_GETPTR1(uniforms,i);
  unsigned char after=uniform<probability?target:before;
  transitions+=after!=before;
  if((after<4)!=(before<4)){
   *(double*)PyArray_GETPTR2(weights,row,col)=after<4?low:high;
   (*(npy_int64*)PyArray_GETPTR2(counts,row,col))++;
   switches++;if(ltd)down_switches++;else up_switches++;
  }
  *state=after;
 }
 return Py_BuildValue("LLLL",transitions,switches,down_switches,up_switches);
}
static PyMethodDef methods[]={{"discrete",discrete,METH_VARARGS,"Apply resolved discrete plasticity with supplied draws."},{"bernoulli",bernoulli,METH_VARARGS,"Exact PCG64 output with independent interleaved recurrence."},{"advance",advance,METH_VARARGS,"Advance IO population with exact NumPy exp/matmul."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_io_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__io_native(void){import_array();PyObject*np=PyImport_ImportModule("numpy");if(!np)return NULL;exp_func=PyObject_GetAttrString(np,"exp");Py_DECREF(np);if(!exp_func)return NULL;return PyModule_Create(&module);}
