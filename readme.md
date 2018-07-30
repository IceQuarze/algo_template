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
### AC自动机
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
### 后缀数组
```C++
char s[N+5];
int sa[N+5],x[N+5],y[N+5],c[N+5];
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
}
int height[N+5];
void SA_H(){
	int i,j,k=0;
	for(i=0;i<len;height[i++]=k)
		for(k>0?--k:0,j=sa[rank[i]-1];s[i+k]==s[j+k];++k);
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
### 并查集
```C++
int tree[N+5];
int aci(int cur){
	if(tree[cur]==0){
		return cur;
	}else{
		return tree[cur]=aci(tree[cur]);
	}
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
### Treap
```C++
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
