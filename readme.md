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
}```
