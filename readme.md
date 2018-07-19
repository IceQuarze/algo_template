## 算法模版
### gcd
```C++
int gcd(int a,int b){
	return b==0?a:gcd(b,a%b);
}
```
### ext_gcd
```C++
int ext_gcd(int a,int b,int &x,int &y){
	int d=a;
	if(b) d=ext_gcd(b,a%b,y,x),y-=(a/b)*x;
	else x=1,y=0;
	return d;
}
```
### KMP
```C++
void build(){
	for(int k=0,q=2;q<=len;++q){
		while(pat[q]!=pat[k+1]&&k>0)
			k=nxt[k];
		if(pat[q]==pat[k+1])
			++k;
		nxt[q]=k;
	}
}
void kmp(){
	for(int k=0,q=1;q<=len;++q){
		while(str[q]!=pat[k+1]&&k>0)
			k=nxt[k];
		if(str[q]==pat[k+1])
			++k;
		if(k==len_pat)
			cout<<"Found"<<endl;
	}
}
```
### 左偏树
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
### spfa(dfs判负环)
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
### 欧拉筛法
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
### 快速乘
```C++
int fm(int a,int b,int m){
	int ans=0;
	while(b){
		if(b&1){
			ans+=a;
			ans%=m;
		}
		b>>=1;
		a+=a;
		a%=m;
	}
	return ans;
}
```
### 快速幂
```C++
int fp(int a,int r,int m){
	int t=a%m,ans=1;
	while(r){
		if(r&1){
			ans*=t;
			ans%=m;
		}
		r>>=1;
		t*=t;
		t%=m;
	}
	return ans;
}
```
### RM素数判断
```C++
//2,3,7,11,61,24251
bool RM(int n,int p){
	int d=n-1;
	while(d%2==0){
		d>>=1;
	}
	for(int i=n-1;i>=d;i>>=1){
		int t=fm(p,r,n);
		if(t==n-1){
			return true;
		}else if(t!=1&&t!=n-1){
			return false;
		}
	}
	return true;
}
```
### RHO素因数分解
```C++
int f(int x,int m){
	int r;
	if(m>1000000){
		r=999979; //魔数
	}else if(m>1000){
		r=997; //魔数
	}else{
		r=rand()%10;
	}
	return (fm(x,x,m)+r)%m;
}
int RHO(int n){
	int a=rand()%n;
	int b=a;
	while(true){
		a=f(a,m);b=f(f(b,m),m);
		if(a==b){
			return -1;
		}
		int g=gcd(abs(a-b),m);
		if(g!=1&&g!=n){
			return g;
		}
	}
}
```
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
