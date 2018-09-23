_____
### 矩阵求逆（模求逆）%%sm|Luogu_4783%%
```C++
int mtx[N+5][2*N+5]
bool guass(){
	int div;
	for(int i=1;i<=n;++i){
		int k=i;
		for(int j=i+1;j<=n;++j) if(mtx[j][i]>mtx[k][i]) k=j;
		if(!(div=mtx[k][i])) return false;
		for(int j=i+1;j<=2*n;++j) swap(mtx[k][j],mtx[i][j]);
		for(int j=i+1,inv=fp(div,MOD-2,MOD);j<=2*n;++j) mtx[i][j]=(mtx[i][j]*inv)%MOD;
		for(k=1;k<=n;++k){
			if(k==i) continue;
			div=mtx[k][i];
			for(int j=i;j<=2*n;++j) mtx[k][j]=((mtx[k][j]-div*mtx[i][j])%MOD+MOD)%MOD;
		}
	}
	return true;
}
```
### 矩阵求逆（浮点数）%%sm|Luogu_3389%%
```C++
double mtx[N+5][2*N+5]
bool gauss(){
	int div;
    for(int i=1;i<=n;i++){
        int k=i;
        for(int j=i+1;j<=n;j++)if(fabs(mtx[j][i])>fabs(mxt[k][i])) k=j;
        if(fabs(div=mtx[k][i])<EPS)return false;
        for(int j=i;j<=2*n;j++)swap(div[k][j],div[i][j]);
        for(int j=i;j<=2*n;j++) mtx[i][j]/=div;
        for(k=1;k<=n;k++)if(k!=i){
                div=mtx[k][i];
                for(int j=i;j<=2*n;j++) mtx[k][j]-=div*mtx[i][j];
            }
    }
    return true;
}
```
_____
### 线性基 %%sm|BZOJ_2460 Luogu_3812%%
```C++
ull p[N+5];
void ins(ull x){
	for(ll i=0;i<=64;++i){
		if(!((x>>i)&1)) continue;
		if(!p[i]){
		/*
			p[i]=x;
			return;	//异或最大
		*/
			
			for(int j=0;j<i;++j) if(x>>j&1LL) x^=p[j];
			for(int j=i+1;j<=64;++j) if(p[j]>>i&1LL) p[j]^=x;
			p[i]=x;
			++cnt;
			return;	//异或第k大
		}else x^=p[i];
	}
}
ull getMX(){
	ull ans=0;
	for(ll i=64;i>=0;--i){
		ans=max(ans,ans^p[i]);
	}
	return ans;
}
bool hasNum(ull x){
	for(ll i=0;i<=64;++i){
		if((x>>i)&1) x^=p[i];
	}
	if(x==0) return true;
	else return false;
}
vector<ll> vb;
void getBase(){
	for(int i=0;i<=ML;++i) if(p[i]) vb.push_back(p[i]);
}
ll getK(ll k){
	ll ans=0;
	for(ll i=0;i<(int)vb.size();++i){
		if(k>>i&1) ans^=vb[i];
	}
	return ans;
}
```
_____
### k短路_A*优化 %%sm|POJ_2449%%
```C++
struct data{
    int g,h,cur;
}dt;
bool operator<(const data &a,const data &b){//大根堆注意
    if(a.g+a.h!=b.g+b.h) return a.g+a.h>b.g+b.h;
    else return a.g>b.g;
}
int AS(){
    priority_queue<data> q;
    int cnt=0;
    if(s==t) ++k;
    dt.cur=s,dt.g=0,dt.h=dis[s];
    q.push(dt);
    while(!q.empty()){
        data dd=q.top();
        q.pop();
        int cur=dd.cur;
        if(cur==t) ++cnt;
        if(cnt==k) return dd.g;
        for(int i=0;i<(int)ve[cur].size();++i){
            int nxt=ve[cur][i].first,w=ve[cur][i].second;
            dt.cur=nxt,dt.g=dd.g+w,dt.h=dis[nxt];
            q.push(dt);
        }
	}
	return -1;//找不到
}
```
_____
### Dijkstra最短路 %%sm|HDU_2544%%
```C++
vector<pair<int,int>> ve[N+5];
int dis[N+5];
void dij(){
	memset(dis,0x3f,sizeof(dis));
	dis[1]=0;
	priority_queue<pair<int,int>> q;
	q.push(make_pair(0,1));
	while(!q.empty()){
		int cur=q.front().second;
		q.pop();
		for(int i=0;i<(int)ve[cur].size();++i){
			int nxt = ve[cur][i].first, w = ve[cur][i].second;
			if (dis[nxt] > dis[cur] + w) {
                dis[nxt] = dis[cur] + w;
                q.push(make_pair(dis[nxt], nxt));
            }
		}
	}
}
```
_____
### Java快速输入输出
```java
import java.util.StringTokenizer;
import java.io.PrintWriter;
StringBuilder sb=new StringBuilder();
while(true){
	int t=System.in.read();
	if(t==-1) break;
	else sb.append((char)t);
}
String s=sb.toString();
StringTokenizer in=new StringTokenizer(s);
PrintWriter out=new PrintWriter(System.out);
```
_____
### NTT快速数论变换 %%sm|Luogu_3803%%
```C++
#define P 998244353LL
#define G 3	//原根
vector<ll> cidx;
ll top,l;
void build(ll sz){
    top=1,l=0;
    while(top<sz) top<<=1,++l;
    top<<=1,++l;
    cidx=vector<ll>(top);
    for(ll i=0;i<top;++i) cidx[i]=(cidx[i>>1]>>1)|((i&1)<<(l-1));
}
void NTT(vector<ll> &v,ll flag){
    for(ll i=0;i<top;++i) if(i<cidx[i]) swap(v[i],v[cidx[i]]);
    for(ll i=1;i<top;i<<=1){
        ll omn=fp(G,(P-1)/(i<<1),P); if(flag==-1) omn=fp(omn,P-2,P);
        for(ll j=0;j<top;j+=(i<<1)){
            ll om=1;
            for(ll k=0;k<i;++k,om=om*omn%P){
                ll omx=v[j+k],omy=om*v[j+i+k]%P;
                v[j+k]=(omx+omy)%P;
                v[j+i+k]=((omx-omy)%P+P)%P;
            }
        }
    }
    if(flag==-1){
        ll inv=fp(v.size(),P-2,P);
        for(ll i=0;i<(ll)v.size();++i) v[i]=v[i]*inv%P;
    }
}
```
### 迭代FFT
```C++
//输出的时候(int)(c.real()+0.5)转成%d输出，%.0f输出-0.1的时候会带上负号
typedef complex<double> _cd;
typedef vector<_cd> _vcd;
int top,l;
void build(int sz){
    top=1,l=0;
    while(top<sz) top<<=1,++l;
    cidx=vector<int>(top);
    for(int i=0;i<top;++i) cidx[i]=(cidx[i>>1]>>1)|((i&1)<<(l-1));
}
void FFT(_vcd &v,int flag){
    for(int i=0;i<top;++i) if(i<cidx[i]) swap(v[i],v[cidx[i]]);
    for(int i=1;i<top;i<<=1){
        _cd omn(cos(M_PI/i),flag*sin(M_PI/i));
        for(int j=0;j<top;j+=(i<<1)){
            _cd om(1,0);
            for(int k=0;k<i;++k,om*=omn){
                _cd omx=v[j+k],omy=om*v[j+i+k];
                v[j+k]=omx+omy;
                v[j+i+k]=omx-omy;
            }
        }
    }
}
```
### 递归FFT
```C++
using _cd=complex<double>;
using _vcd=vector<_cd>;
_vcd FFT(_vcd &a,int sign){ //sign=1 FFT,sign=-1 IFFT
	int n=a.size();
	auto omn=_cd(cos(sign*2*M_PI/n),sin(2*M_PI/n));
	auto om=_cd(1,0);
	_vcd a0,a1;
	for(int i=0;i<n;++i){
		if(i%2==0) a0.push_back(a[i]);
		else a1.push_back(a[i]);
	}
	_vcd y1,y2;
	y0=FFT(a0);
	y1=FFT(a1);
	_vcd y(n);
	for(int i=0;i<n/2;++i){
		y[i]=y0[i]+om*y1[i];
		y[i+n/2]=y0[i]-om*y1[i];
		om=om*omn;
	}
	return y;
}
```
_____
### 树状数组
```C++
int tree[N+5];
void add(int i,int x){
	while(i<=n){
		tree[i]+=x;
		i+=i&-i;
	}
}
int get(int i){
	int sum=0;
	while(i>0){
		sum+=tree[i];
		i-=i&-i;
	}
	return i;
}
```
_____
### 快速排序
```C++
void qs(vector<int> &v,int l,int r){
	if(l>=r) return;
	int pl=l,pr=r;
	int mv=v[(l+r)/2];
	while(pl<=pr){
		while(v[pl]<mv) ++pl;
		while(v[pr]>mv) --pr;
		if(pl<=pr){
			swap(v[pl],v[pr]);
			++pl,++pr;
		}
	}
	qs(v,l,pr);
	qs(v,pl,r);
}
		
```
_____
### 小根堆
```C++
int heap[N+5],top=0;
void ins(int v){
	heap[++top]=v;
	int po=top;
	while(po>1&&heap[po]<heap[po>>1])
		swap(heap[po],heap[po>>1]),po>>=1;
}
void pop(){
	swap(heap[top],heap[1]);
	heap[top--]=0;
	int po=2;
	while(po<=top){
		if(po<top&&heap[po+1]<heap[po]) ++po;
		if(heap[po]<heap[po>>1]) swap(heap[po],heap[po>>1]),po<<=1;
		else break;
	}
}
```
_____
### GDB pretty printer
```python
python
import sys
sys.path.insert(0, 'C:/Program Files (x86)/CodeBlocks/MinGW/share/gcc-5.1.0/python/libstdcxx/v6')
from printers import register_libstdcxx_printers
register_libstdcxx_printers (None)
end
```
_____
### gcd
```C++
int gcd(int a,int b){
	return b==0?a:gcd(b,a%b);
}
```
_____
### ext_gcd
```C++
int ext_gcd(int a,int b,int &x,int &y){
	int d=a;
	if(b) d=ext_gcd(b,a%b,y,x),y-=(a/b)*x;
	else x=1,y=0;
	return d;
}
```
### 费马小定理
```C++
$$X=A^{p-2}$$
```
### 线性递推逆元 %%sm|Luogu_3811%%
```C++
/*
设$$p=k*i+r$$
$$k∗i+r≡0(mod\ p)$$
$$k*r^{-1}+i^{-1}≡0(mod\ p)$$
$$i^{-1}≡-k*r^{-1}(mod\ p)$$
$$i^{-1}≡(-\lfloor \frac{p}{i}\rfloor *(p\ mod\ i))\ (mod\ p)$$
*/
int dp[N+5];
dp[1]=1;
for(int i=2;i<=n;++i){
	dp[i]=-(p/i)*dp[p%i];
	dp[i]=(dp[i]%p+p)%p;
}
```
_____
### KMP %%sm|Luogu_3375%%
```C++
void build(){
	nxt[1]=-1;
	for(int k=-1,q=1;q<len;++q){
		while(pat[q]!=pat[k+1]&&k!=-1)
			k=nxt[k];
		if(pat[q]==pat[k+1])
			++k;
		nxt[q]=k;
	}
}
void kmp(){
	for(int k=-1,q=0;q<len;++q){
		while(str[q]!=pat[k+1]&&k!=-1)
			k=nxt[k];
		if(str[q]==pat[k+1])
			++k;
		if(k==len_pat-1)
			cout<<"Found"<<endl;
	}
}
```
_____
### 扩展KMP %%sm|HDU_2594%%
```C++
int nxt[N+5],ex[N+5];
char s[N+5],pat[N+5];
void build(){
	memset(nxt,0,sizeof(nxt));
	int i,j,po,len=strlen(pat);
	nxt[0]=len;
	for(i=0;i+1<len&&pat[i]==pat[i+1];++i);
	nxt[1]=i;
	for(i=2,po=1;i<len;++i){
		if(i+nxt[i-po]<po+nxt[po])
			nxt[i]=nxt[i-po];
		else{
			j=po+nxt[po]-i;
			if(j<0) j=0;
			for(;i+j<len&&pat[j]==pat[i+j];++j);
			nxt[i]=j;
			po=i;
		}
	}
}
void exkmp(){
	int i,j,po,len1=strlen(s),len2=strlen(pat);
	for(i=0;i<len1&&i<len2&&s[i]==pat[i];++i);
	ex[0]=i;
	for(i=1,po=0;i<len1;++i){
		if(i+nxt[i-po]<p+ex[po])
			ex[i]=nxt[i-po];
		else{
			j=po+ex[po]-i;
			if(j<0) j=0;
			for(;i+j<len1&&j<len2&&s[i+j]==pat[j];++j);
			ex[i]=j;
			po=i;
		}
	}
}
```
_____
### AC自动机 %%sm|Luogu_3808%%
```C++
struct node{
	struct node *nxt[26],*fail;
	int cnt;
	bool isE;
	node(){
		memset(nxt,0,sizeof(nxt));
		fail=NULL;
		cnt=0;
		isE=false;
	}
};
void ins(struct node *cur,const char *s){
	if(*s=='\0'){
		isE=true;
		return;
	}
	if(cur->nxt[*s-'a']==NULL) cur->nxt[*s-'a']=new node();
	ins(cur->nxt[*s-'a'],s+1);
}
void build(){
	queue<struct node *> q;
	q.push(root);
	root->fail=NULL;
	while(!q.empty()){
		struct node *cur=q.front();q.pop();
		for(int i=0;i<26;++i){
			if(cur->nxt[i]==NULL) continue;
			struct node *f=cur->fail;
			while(true){
				if(f==NULL){
					cur->nxt[i]->fail=root;
					break;
				}
				if(f->nxt[i]!=NULL&&f->nxt[i]!=cur->nxt[i]){
					cur->nxt[i]->fail=f->nxt[i];
					break;
				}
				else{
					f=f->fail;
				}
			}
			q.push(cur->nxt[i]);
		}
	}
}
void ac(const char *s){
	struct node *cur=root;
	for(int i=0;s[i]!='\0';++i){
		while(cur!=root&&cur->nxt[s[i]-'a']==NULL) cur=cur->fail;
		if(cur->nxt[s[i]-'a']!=NULL){
			cur=cur->nxt[s[i]-'a'];
			struct node *p=cur;
			while(p!=root){ //如果不求每个单词出现的数量，遇到之前访问过的字符串时，直接退出
				++p->cnt;
				p=p->fail;
			}
		}
	}
}
map<string,int> mp;
string stk;
void get(int cur){
	if(cur->isE==true){
		++mp[stk];
		return;
	}
	for(int i=0;i<26;++i){
		if(cur->nxt[i]!=NULL){
			stk.push_back('a'+i);
			get(cur->nxt[i]);
			stk.pop_back();
		}
	}
}
void del(struct node *cur){
	if(cur==NULL) return;
	for(int i=0;i<26;++i) del(cur->nxt[i]);
	delete cur;
}
```
_____
### 后缀数组 %%sm|POJ_2774%%
```C++
char s[N+5];
int sa[N+5],x[N+5],y[N+5],c[N+5]; // 字符串下标 0~len-1 , 排名 0~len-1
int len,m;
void SA(){
	len=strlen(s),m=128;
	memset(c,0,(m+1)*sizeof(int));
	for(int i=0;i<len;++i) x[i]=s[i];
	for(int i=0;i<len;++i) ++c[x[i]];
	for(int i=2;i<=m;++i) c[i]+=c[i-1];
	for(int k=1;k<=len;++k){
		int clk=0;
		for(int i=len-k;i<len;++i) y[clk++]=i;
		for(int i=0;i<len;++i) if(sa[i]>=k) y[clk++]=sa[i]-k;
		memset(c,0,(m+1)*sizeof(int));
		for(int i=0;i<len;++i) ++c[x[i]];
		for(int i=2;i<=m;++i) c[i]+=c[i-1];
		for(int i=len-1;i>=0;--i) sa[--c[x[y[i]]]=y[i],y[i]=0;
		swap(x,y);
		clk=1;x[sa[0]]=1;
		for(int i=1;i<len;++i){
			if(y[sa[i]]!=y[sa[i-1]]||y[sa[i]+k]!=y[sa[i-1]+k])
				x[i]=++clk;
			else
				x[i]=clk;
		}
		if(clk==len) break;
		m=clk;
	}
	for(int i=0;i<len;++i)
		x[sa[i]]=i;
}
int height[N+5];
void SA_H(){
	int i,j,k=0;
	for(i=0;i<len;height[rank[i++]]=k)
		for(k>0?--k:0,j=sa[rank[i]-1];s[i+k]==s[j+k];++k);
}
```
_____
### 左偏树 %%sm|HDU_1512 Luogu_3377%%
```C++
node* merge(node *a,node *b){
	if(a==NULL)
		return b;
	if(b==NULL)
		return a;
	if(a->v<b->v)
		swap(a,b);
	a->r=merge(a->r,b);
	if(a->l==NULL||a->l->h<a->r->h)
		swap(a->l,a->r);
	a->h=a->l->h+1;
	return a;
}
```
_____
### 并查集 %%sm|Luogu_3367%%
```C++
int tree[N+5];
int aci(int cur){
	if(tree[cur]==0) return cur;
	else return tree[cur]=aci(tree[cur]);
}
```
_____
### tarjan缩点
```C++
vector<int> ve[N];
vector<int> stk;
int mark[N];
int dfn[N],low[N];
int clk=0,mclk=0;
void tarjan(int cur){
	stk.push_back(cur);
	dfn[cur]=low[cur]=++clk;
	for(auto &nxt:ve[cur]){
		if(!dfn[nxt]){
			tarjan(nxt);
			low[cur]=min(low[cur],low[nxt]);
		}else if(!mark[nxt]){
			low[cur]=min(low[cur],dfn[nxt]);
		}
	}
	if(low[cur]==dfn[cur]){
		++mclk;
		int t;
		while(true){
			t=stk.back();
			mark[t]=mclk;
			if(t==cur)
				break;
		}
	}
}
```
_____
### tarjan最近公共祖先
``` C++
void tarjan(int cur){
	for(auto &nxt:ve[cur]){
		tarjan(nxt);
		merge(cur,nxt);
		vst[nxt]=true;
	}
	for(auto &q:query[cur]){
		if(vst[q]){
			ans.push_back(aci(q));
		}
	}
}
```
_____
### spfa(dfs判负环) %%sm|Luogu_P3385%%
```C++
vector<int> ve[N];
int vst[N],dis[N],w[N][N];
bool flag=false;
//dis[cur]=0;
void spfa(int cur){
	vst[cur]=true;
	for(auto &nxt:ve[cur]){
		if(dis[nxt]>dis[cur]+w[cur][nxt]){
			if(vst[nxt]){
				flag=true;
				return;
			}
			dis[nxt]=dis[cur]+w[cur][nxt];
			spfa(nxt);
		}
	}
	vst[cur]=false;
}
```
_____
### spfa(bfs最短路)
```C++
vector<int> ve[N]
int vst[N],dis[N],w[N][N],cnt[N];
void spfa(int root){
	queue<int> q;
	q.push(root);
	dis[root]=0;
	++cnt[root];
	vst[root]=true;
	while(!q.empty()){
		int cur=q.front();q.pop();
		vst[cur]=false;
		if(cnt[cur]>n){
			flag=true;
			return;
		}
	
		for(auto &nxt:ve[cur]){
			if(dis[nxt]>dis[cur]+w[cur][nxt]){
				dis[nxt]=dis[cur]+w[cur][nxt];
				if(!vst[nxt]){
					q.push(nxt);
					vst[nxt]=true;
					++cnt[nxt];
				}
			}
		}
	}
}
```
_____
### 埃氏筛法
```C++
bool nP[N];
void SI(){
	for(int i=2;i<=N;++i){
		if(nP[i]) continue;
		for(int j=2;i*j<=N;++j)
			nP[i*j]=true;
	}
}
```
_____
### 欧拉筛法 %%sm|Luogu_3383%%
```C++
vector<int> vp;
int vst[N];
void SI(){
	for(int i=2;i<=N;++i){
		if(!vst[i]) vp.push_back(i);
		for(auto &p:vp){
			if(p*i>N) break;
			vst[p*i]=true;
			if(i%p==0) break;
		}
	}
}
```
_____
### 快速乘
```C++
int fm(int a,int b,int m){
	int t=a%m,ans=0;
	while(b){
		if(b&1){
			ans=ans+t;
			if(ans>m) ans-=m;	//取模速度可能会慢一倍
		}
		t=t+t;
		if(t>m) t-=m;
		b>>=1;
	}
	return ans;
}
```
_____
### 快速幂
```C++
int fp(int a,int r,int m){
	int t=a%m,ans=1;
	while(r){
		if(r&1) ans=(ans*t)%m;
		t=(t*t)%m;
		r>>=1;
	}
	return ans;
}
```
_____
### RM素数判断
```C++
//2,3,7,11,61,24251
bool RM(int n,int p){
    for(int i=n-1;;i>>=1){
        int t=fp(p,i,n);
        if(t==n-1) return true;
        else if(t!=1&&t!=n-1) return false;
        if(i&1) return true;
    }
}
```
_____
### RHO素因数分解 %%sm|POJ 1811%%
```C++
int f(int x,int m){
	int r;
	if(m>1e6) r=1e6;
	else r=rand()%10;
	return (fm(x,x,m)+r)%m;
}
int RHO(int n){
	int a=rand()%10;int b=a;	//初始化成一个随机数很重要
	while(true){
		a=f(a,n);b=f(f(b,m),n);
		if(a==b) return -1;
		int g=gcd(abs(a-b),n);
		if(g!=1&&g!=n) return g;
	}
}
```
_____
### 欧拉路
```C++
int cnt[N][N];
vector<int> pth;
void dfs(int cur){
    for(int i=0;i<=6;++i){
        while(cnt[cur][i]){
            --cnt[cur][i];
            --cnt[i][cur];
            dfs(i);
        }
    }
    pth.push_back(cur);
}
```
_____
### 尼姆博奕sg函数
```C++
vector<int> nxt;
int sg(int n){
	if(n==0){
		return 0;
	}else{
		vector<bool> vst(N+5,false);
		for(auto &t:nxt){
			if(t>n){
				break;
			}else{
				vst[n-nxt]=true;
			}
		}
		for(int i=0;i<=N;++i){
			if(!vst[i]){
				return i;
			}
		}
	}
}
```
_____
### Treap
```C++
inline int random(){
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}
struct node{
	int prio,v;
	node *l,*r;
}
node *rrot(node *cur){
	node *nxt=cur->l;
	cur->l=nxt->r;
	nxt->r=cur;
	return nxt;
}
node *lrot(node *cur){
	node *nxt=cur->r;
	cur->r=nxt->l;
	nxt->l=cur;
	return nxt;
}
node *ins(node *cur,node *tp){
	if(cur==NULL){
		return tp;
	}
	if(cur->v>tp->v){
		cur->l=ins(cur->l,tp);
		if(cur->prio>cur->l->prio){
			cur=rrot(cur);
		}
	}else{
		cur->r=ins(cur->r,tp);
		if(cur->prio>cur->r->prio){
			cur=lrot(cur);
		}
	}
	return cur;
}
node *rmv(node *cur,int v){
	if(cur==NULL){
		return NULL;
	}
	if(cur->v>v){
		cur->l=rmv(cur->l,v);
	}else if(cur->v<tp->v){
		cur->r=rmv(cur->r,v);
	}else{
		if(cur->l==NULL){
			return cur->r;
		}else if(cur->r==NULL){
			return cur->l;
		}
		if(cur->l->prio<cur->r->prio){
			cur=rrot(cur);
			cur->r=rmv(cur->r,v);
		}else{
			cur=lrot(cur);
			cur->l=rmv(cur->l,v);
		}
	}
	return cur;
}
```
_____
### 欧拉函数
``` C++
int phi(int n){
	int mul=n;
	for(int i=2;i*i<=n;++i){
		if(n%i){
			mul-=mul/i;
			while(n%i==0) n/=i;
		}
	}
	if(n>1){
		mul-=mul/n;
	}
	return mul;
}
```
_____
### 区间线段树
```C++
int tree[N+5],inc[N+5];
void udp(int cur,int l,int r,int tl,int tr,int v){
    int mid=(r+l)/2;
    if(l==tl&&r==tr){
        tree[cur]+=(tr-tl+1)*v;
        inc[cur]+=v;
        return;
    }
    if(tr<=mid){
        udp(cur*2,l,mid,tl,tr,v);
    }else if(tl>=mid+1){
        udp(cur*2+1,mid+1,r,tl,tr,v);
    }else{
        udp(cur*2,l,mid,tl,mid,v);
        udp(cur*2+1,mid+1,r,mid+1,tr,v);
    }
    tree[cur]+=(tr-tl+1)*v;
}
int get(int cur,int l,int r,int tl,int tr){
    int mid=(r+l)/2;
    if(l==tl&&r==tr){
        return tree[cur];
    }
    tree[cur*2]+=(mid-l+1)*inc[cur];
    tree[cur*2+1]+=(r-(mid+1)+1)*inc[cur];
    inc[cur*2]+=inc[cur];
    inc[cur*2+1]+=inc[cur];
    inc[cur]=0;
    if(tr<=mid){
        return get(cur*2,l,mid,tl,tr);
    }else if(tl>=mid+1){
        return get(cur*2+1,mid+1,r,tl,tr);
    }else{
        return get(cur*2,l,mid,tl,mid)+get(cur*2+1,mid+1,r,mid+1,tr);
    }
}
```
_____
### 马拉车回文算法
```C++
//s="$#a#b#c#d#"
int p[N+5];
int mlc(){
	int len=strlen(s);
	int mx,mx_len,id;
	for(int i=1;i<len;++i){
		if(i<mx){
			p[i]=min(p[2*id-i],mx-i);
		}else{
			p[i]=1;
		}
		while(s[i-p[i]]==s[i+p[i]]){
			++p[i];
		}
		if(i+p[i]>mx){
			id=i;
			mx=i+p[i];
		}
		mx_len=max(mx_len,p[i]-1);
	}
	return mx_len;
}
```
_____
### st表
```C++
int tree[N+5][25];
int num[N+5];
void ST(){
	for(int i=1;i<=N;++i){
		tree[i][0]=num[i];
	}
	for(int j=1;j<=20;++j){
		for(int i=1;i<=N;++i){
			if(i+(1<<(j-1)<=n)
				tree[i][j]=max(tree[i][j-1],tree[i+(1<<(j-1)][j-1]);
		}
	}
}
int cal(int n){
int cnt=0;
	while(n){
		++cnt;
		n>>=1;
	}
	return cnt;
}
int search(int l,int r){
	int d=r-l+1;
	int c=cal(d);
	return max(tree[l][c],tree[r-(1<<c)+1][c]);
}
```
_____
### 匈牙利算法|最大二分图匹配
```C++
vector<int> ve[N+5];
bool vst[N+5];
int usd[N+5];
bool find(int left){
	for(int i=0;i<(int)ve[left].size();++i){
		int nxt=ve[left][i];
		if(!vst[nxt]){
			vst[nxt]=true;
			if(usd[nxt]==0||find(usd[nxt]){
				usd[nxt]=left;
				return true;
			}
		}
	}
	return false;
}
int main(){
	int cnt=0;
	for(int i=1;i<=n;++i){
		if(find(i)) ++cnt;
	}
}
```
_____
### 最长公共上升子序列
```C++
//a[i]!=b[j]:   F[i][j]=F[i-1][j]
//a[i]==b[j]:   F[i][j]=max(F[i-1][k])+1 1<=k<=j-1&&b[j]>b[k]
void solve(){
	for(int i=1;i<=n;++i){
		int mx=0;
		for(int j=1;j<=m;++j){
			if(a[i]>b[i]) mx=max(mx,dp[i-1][j]);
			if(a[i]!=b[i]) dp[i][j]=dp[i-1][j];
			else dp[i][j]=max+1;
		}
	}
}
```
_____
### 数位dp
```C++
vector<int> v;
int dp[10][10];
int cal(int dep){
    int sum=0;

    for(int i=0;i<=9;++i) dp[dep][i]=0;
    for(int i=0;i<=9;++i){
        dp[dep][v[dep]]+=dp[dep-1][i];
    }
    if(不能转移的情况去除){
    }
    if(dep==v.size()-1){
        sum+=dp[dep][v[dep]];
    }else{
        sum+=cal(dep+1);
    }

    for(int i=0;i<=9;++i) dp[dep][i]=0;
    for(int i=0;i<v[dep];++i){
        for(int j=0;j<=9;++j){
            dp[dep][i]+=dp[dep-1][j];
        }
        if(不能转移的情况去除){
    	}
    }
    if(dep==v.size()-1){
        for(int i=0;i<v[dep];++i) sum+=dp[dep][i];
    }else{
        v[dep+1]=9;
        sum+=cal(dep+1);
    }

    return sum;
}
```
_____
### End
