"""PIT and Q-Q animations: a simulated example and the real crisis-period FHS-GJR forecasts."""
import numpy as np
from scipy.stats import johnsonsu,norm,t
from manim import *
import manimpango
from theme import *

# TeX Live's fonts are not necessarily installed as system fonts; the date label uses Text.
manimpango.register_font(str(ROOT/'assets/lmroman10-regular.otf'))

CRISIS_START='2018-01-01'
CRISIS_END='2021-12-31'
GOLD='#909090'
QQ_X_LIMIT=4
QQ_Y_LIMIT=12
QQ_INFINITY_OFFSET=.85


def crisis_mask(data):
    return (data['dates']>=CRISIS_START)&(data['dates']<=CRISIS_END)


def forecast_scores(data,prefix,pick):
    """Raw PITs; returns outside the forecast support get infinite Normal scores."""
    if prefix=='normal':
        return data['normal_u'][pick],data['normal_z'][pick]
    u=data[prefix+'_raw_u'][pick]
    return u,norm.ppf(u)


def simulated_pit(n=240,seed=42):
    """Returns drawn from changing true laws, scored with nearby (not exact) forecast laws."""
    time=np.arange(n)
    a=.95*np.sin(.4*time)+.25*np.cos(.17*time)
    b=1.35+.25*np.cos(.29*time)
    mean=.65*np.sin(.31*time)
    sd=.9+.18*np.cos(.23*time)

    def law(shape_a,shape_b,average,deviation):
        base_mean,base_var=johnsonsu.stats(shape_a,shape_b,moments='mv')
        scale=deviation/np.sqrt(base_var)
        return johnsonsu(shape_a,shape_b,loc=average-scale*base_mean,scale=scale)

    truth=law(a,b,mean,sd)
    forecast=law(a+.045*np.sin(.71*time+.4),b+.035*np.cos(.47*time),
        mean+.025*np.cos(.53*time),sd*(1+.02*np.sin(.59*time)))
    returns=truth.rvs(random_state=np.random.default_rng(seed))
    uniform=forecast.cdf(returns)
    grid=np.linspace(-8,8,401)
    return dict(returns=returns,uniform=uniform,scores=norm.ppf(uniform),grid=grid,
        forecast_cdf=forecast.cdf(grid[:,None]).T,
        true_pdf=truth.pdf(grid[:,None]).T,forecast_pdf=forecast.pdf(grid[:,None]).T)


def flight_state(points,alpha):
    """Allocate travel by distance, with a short, equal pause at each projection."""
    distances=np.linalg.norm(np.diff(points,axis=0),axis=1)
    travel=.76*distances/max(float(distances.sum()),1e-9)
    duration=travel+.24/len(distances)
    edges=np.r_[0.,np.cumsum(duration)]
    phase=float(np.clip(alpha,0,1))
    k=min(int(np.searchsorted(edges,phase,side='right')-1),len(distances)-1)
    local=(phase-edges[k])/duration[k]
    moving=travel[k]/duration[k]
    eased=smooth(min(local/max(moving,1e-9),1.))
    position=points[k]+eased*(points[k+1]-points[k])
    pulse=float(np.sin(np.pi*np.clip((local-moving*.85)/max(1-moving*.85,1e-9),0,1))**2)
    return position,pulse,k


class PITParticle(VGroup):
    def __init__(self,radius=.043,color=NAVY):
        self.base_color=ManimColor(color)
        self.core=Dot(radius=radius,color=color)
        self.rings=VGroup(*[Circle(radius=radius*f,stroke_color=GOLD,
            stroke_width=w,fill_color=GOLD,fill_opacity=.04).set_opacity(0)
            for f,w in ((2.,1.6),(2.9,1.),(3.8,.6))])
        super().__init__(self.rings,self.core)

    def seek(self,position,pulse,opacity=1):
        self.move_to(position)
        self.core.set_opacity(opacity).set_color(interpolate_color(self.base_color,ManimColor(GOLD),pulse))
        for ring,strength in zip(self.rings,(.8,.35,.13)):
            ring.set_opacity(strength*pulse*opacity)
        return self


class PITTimeline(VGroup):
    """Progress line under the PIT panels, with year marks and the current date."""
    left_x=-6.45
    right_x=6.45
    baseline=-3.55

    def __init__(self,dates,n):
        super().__init__()
        self.dates=dates
        self.position=0.
        self.compact=False
        self.displayed_date=None
        start=[self.left_x,self.baseline,0]
        self.track=Line(start,[self.right_x,self.baseline,0],color=RULE,stroke_width=1.4)
        self.fill=Line(start,[self.left_x+.0001,self.baseline,0],color=NAVY,stroke_width=2.2)
        self.head=Dot(start,radius=.052,color=NAVY)
        self.marks=VGroup()
        def mark(x,text):
            self.marks.add(Line([x,self.baseline-.055,0],[x,self.baseline+.055,0],color=INK,stroke_width=.9),text.move_to([x,-3.95,0]))
        if dates is None:
            mark(self.left_x,eq('t=1',20));mark(self.right_x,eq('t=T',20))
            self.date_label=VGroup()
        else:
            for year in range(int(dates[0][:4]),int(dates[-1][:4])+1):
                index=int(np.searchsorted(dates,f'{year}-01-01'))
                mark(self.left_x+(self.right_x-self.left_x)*index/(n-1),label(str(year),18))
            self.date_label=self.make_date(0)
        self.add(self.track,self.marks,self.fill,self.head,self.date_label)
        self.seek(0,0)

    def make_date(self,index):
        value=str(self.dates[index])[:7] if self.compact else str(self.dates[index])
        self.displayed_date=value
        return Text(value,font='Latin Modern Roman 10',font_size=20,color=NAVY,disable_ligatures=True)

    def seek(self,progress,index=None):
        self.position=float(max(0,min(1,progress)))
        x=self.left_x+(self.right_x-self.left_x)*self.position
        self.fill.put_start_and_end_on([self.left_x,self.baseline,0],[max(x,self.left_x+.0001),self.baseline,0])
        self.head.move_to([x,self.baseline,0])
        if self.dates is not None:
            if index is not None:
                value=str(self.dates[index])[:7] if self.compact else str(self.dates[index])
                if value!=self.displayed_date:self.date_label.become(self.make_date(index))
            self.date_label.move_to([0,-3.23,0])


def qq_infinity_rows(ax,x_limit=QQ_X_LIMIT,y_limit=QQ_Y_LIMIT,size=16):
    """Labelled margin rows for infinite scores."""
    result=VGroup()
    for sign in (-1,1):
        y=sign*(y_limit+QQ_INFINITY_OFFSET)
        result.add(DashedLine(ax.c2p(-x_limit,y),ax.c2p(x_limit,y),color=RULE,
            stroke_width=.65,dash_length=.045),
            eq(r'-\infty' if sign<0 else r'+\infty',size).next_to(ax.c2p(-x_limit,y),LEFT,buff=.14))
    return result


def qq_cloud(ax,x,y,empirical=True,y_limit=5):
    """Finite scores keep exact coordinates; infinite scores are open circles on the margin rows."""
    emphasized=set(np.argmin(abs(x[:,None]-np.linspace(-3.5,3.5,65)),axis=0).tolist())
    result=VGroup()
    for i,(a,b) in enumerate(zip(x,y)):
        if np.isinf(b):
            marker=Circle(radius=.038,stroke_color=NAVY,stroke_width=1.25,
                fill_opacity=0).move_to(ax.c2p(float(a),float(np.sign(b))*(y_limit+QQ_INFINITY_OFFSET)))
        else:
            marker=Dot(ax.c2p(float(a),float(b)),
                radius=(.025 if i in emphasized else .018) if empirical else (.034 if abs(a)>1.645 else .021),
                color=NAVY if i in emphasized or abs(a)>1.645 else INK,
                fill_opacity=(1 if i in emphasized else .55) if empirical else 1)
        result.add(marker)
    return result


class PITAnimation:
    def pit_demo(self,empirical):
        if empirical:
            data=np.load(ROOT/'data/forecasts.npz')
            pick=crisis_mask(data)
            xs=100*data['returns'][pick];us,zs=forecast_scores(data,'gjr',pick)
            dates=data['dates'][pick];support=100*data['gjr_support'][pick];weights=data['gjr_counts'][pick]
            xr=[-13,10,5];title='FHS-GJR · Crisis PIT · 2018–2021'
        else:
            simulation=simulated_pit()
            xs=simulation['returns'];us=simulation['uniform'];zs=simulation['scores']
            dates=None;xr=[-8,8,4];title='PIT · Changing predictive distributions'
        n=len(xs)
        self.frame('Empirical results' if empirical else 'Methods',title)
        # Top: forecast CDF -> uniform PIT -> inverse Normal. Bottom: forecast density and Normal-score histogram.
        cdf=axes(xr,[0,1,.5],4.55,2.15,-7.0,.55)
        den=axes(xr,[0,1,.2],4.55,1.35,-7.0,-2.35)
        pit=axes([0,n*.23,1],[0,1,.5],2.05,2.15,-1.23,.55)
        normal=axes([-5,5,1],[0,1,.5],4.35,2.15,2.8,.55)
        score=axes([-5,5,1],[0,n*.28,1],4.35,1.35,2.8,-2.35)
        for ax in (cdf,den,pit,normal,score):ax.y_axis.set_opacity(0)
        baseline=Line(cdf.c2p(xr[0],0),cdf.c2p(xr[0],1),color=MUTED,stroke_width=.8)
        headings=VGroup(eq(r'\widehat F_{t\mid t-1}(r)',29).move_to([-4.72,3.02,0]),
            eq(r'u_t=\widehat F_{t\mid t-1}(r_t)',27).move_to([-.2,3.02,0]),
            eq(r'z_t=\Phi^{-1}(u_t)',29).move_to([4.98,3.02,0]),
            label('FHS-GJR · Forecast density' if empirical else 'Return densities',21,NAVY).move_to([-4.72,-2.94,0]),
            eq(r'\text{Normal scores}\quad N(0,1)',23).move_to([4.98,-2.94,0]),
            eq(r'U(0,1)',23).move_to([-.2,.12,0]))
        density_key=VGroup() if empirical else VGroup(
            legend_item('True distribution (unobserved)',color=MUTED,size=17),
            legend_item('Predicted distribution',kind='dash',color=ACCENT_COOL,size=17)
        ).arrange(DOWN,buff=.18,aligned_edge=LEFT).move_to([.1,-2.18,0])
        legend=note('t: day; r: realised return; u: forecast percentile; z: Normal score; Phi: standard Normal CDF.',y=3.39,size=17)
        support_note=label('Each dot uses its own dated forecast.' if empirical else
            'Simulated returns',16,MUTED).move_to([.1,-1.35,0])
        ticks=VGroup()
        for value in (0,.5,1):ticks.add(label(f'{value:g}',16,MUTED).next_to(cdf.c2p(xr[0],value),LEFT,buff=.1))
        for value in (-5,0,5):ticks.add(label(str(value),16,MUTED).next_to(score.c2p(value,0),DOWN,buff=.1))
        for value in ((-10,0,10) if empirical else (-6,0,6)):
            ticks.add(label(f'{value:g}'+('%' if empirical else ''),16,MUTED).next_to(den.c2p(value,0),DOWN,buff=.1))
        phi=normal.plot(lambda z:float(norm.cdf(z)),color=INK,stroke_width=1.7)
        # Cumulative histogram counts after each observation.
        uedges=np.linspace(0,1,16);zedges=np.linspace(-5,5,21)
        zbin=zedges.copy();zbin[[0,-1]]=[-np.inf,np.inf]
        ubins=np.clip((us*15).astype(int),0,14)
        zbins=np.clip(np.searchsorted(zbin,zs,side='right')-1,0,19)
        uh=np.zeros((n+1,15),int);zh=np.zeros((n+1,20),int)
        for i,(u,z) in enumerate(zip(ubins,zbins)):
            uh[i+1]=uh[i];zh[i+1]=zh[i];uh[i+1,u]+=1;zh[i+1,z]+=1
        def histograms(count,blend=0.):
            result=VGroup()
            uniform=(1-blend)*uh[count]+blend*uh[min(n,count+1)]
            scores=(1-blend)*zh[count]+blend*zh[min(n,count+1)]
            for lo,hi,h in zip(uedges[:-1],uedges[1:],uniform):
                p0,p1=pit.c2p(0,lo),pit.c2p(max(float(h),.001),hi)
                result.add(Rectangle(width=p1[0]-p0[0],height=(p1[1]-p0[1])*.88,stroke_width=0,fill_color=NAVY,fill_opacity=.6).move_to((p0+p1)/2))
            for lo,hi,h in zip(zedges[:-1],zedges[1:],scores):
                p0,p1=score.c2p(lo,0),score.c2p(hi,max(float(h),.001))
                result.add(Rectangle(width=(p1[0]-p0[0])*.87,height=p1[1]-p0[1],stroke_width=0,fill_color=NAVY,fill_opacity=.6).move_to((p0+p1)/2))
            return result
        hist=histograms(0)
        target=VGroup(DashedLine(pit.c2p(n/15,0),pit.c2p(n/15,1),color=MUTED,stroke_width=1,dash_length=.07),
            DashedVMobject(score.plot(lambda z:float(n*(norm.cdf(z+.25)-norm.cdf(z-.25))),color=MUTED,stroke_width=1),num_dashes=35))
        def forecast(i):
            if not empirical:
                def curve(ax,values,color,width=1.7):
                    return VMobject(stroke_color=color,stroke_width=width).set_points_as_corners([
                        ax.c2p(float(x),float(y)) for x,y in zip(simulation['grid'],values)])
                density=curve(den,simulation['true_pdf'][i]/1.2,MUTED,3.0)
                area=VMobject(stroke_width=0,fill_color='#A0A0A0',fill_opacity=.22)
                area.set_points_as_corners([den.c2p(xr[0],0),
                    *[den.c2p(float(x),float(y)/1.2) for x,y in zip(simulation['grid'],simulation['true_pdf'][i])],
                    den.c2p(xr[1],0)]).close_path()
                dashes=[]
                values=simulation['forecast_pdf'][i]/1.2
                for j in range(0,len(values)-7,12):
                    piece=VMobject().set_points_as_corners([
                        den.c2p(float(x),float(y)) for x,y in zip(simulation['grid'][j:j+7],values[j:j+7])])
                    dashes.append(piece.get_points())
                predicted=VMobject(stroke_color=ACCENT_COOL,stroke_width=4.0).set_points(np.concatenate(dashes))
                predicted.set_stroke(BG,width=6.0,background=True)
                return VGroup(curve(cdf,simulation['forecast_cdf'][i],NAVY),
                    area,density,predicted)
            # FHS forecast: 250 standardised shocks weighted by their 10,000 resample counts.
            order=np.argsort(support[i]);s=support[i,order];w=weights[i,order]
            cumulative=np.cumsum(w)
            before=float(w[s<xr[0]].sum())/10000
            points=[cdf.c2p(xr[0],before)]
            for value,count in zip(s,cumulative):
                if xr[0]<=value<=xr[1]:
                    after=float(count)/10000
                    points.extend([cdf.c2p(float(value),before),cdf.c2p(float(value),after)])
                    before=after
            points.append(cdf.c2p(xr[1],before))
            step=VMobject(stroke_color=NAVY,stroke_width=1.6).set_points_as_corners(points)
            edges=np.linspace(xr[0],xr[1],46)
            heights=np.histogram(s,bins=edges,weights=w/10000)[0]
            bars=VGroup()
            for lo,hi,h in zip(edges[:-1],edges[1:],heights):
                p0,p1=den.c2p(lo,0),den.c2p(hi,min(float(h)/.8,1))
                bars.add(Rectangle(width=(p1[0]-p0[0])*.92,height=max(.001,p1[1]-p0[1]),stroke_width=0,fill_color=NAVY,fill_opacity=.68).move_to((p0+p1)/2))
            return VGroup(step,bars)
        laws=None if empirical else [forecast(i) for i in range(n)]
        forecast_mob=forecast(0) if empirical else laws[0].copy()
        timeline=PITTimeline(dates,n)
        plot_axes=VGroup(cdf,den,pit,normal,score,baseline)
        self.add(plot_axes,headings,legend,ticks,phi,target,forecast_mob,timeline,support_note,density_key,hist)
        # Static introduction: wait for the presenter before starting the animation.
        self.finish('empirical_pit_setup' if empirical else 'correct_pit_setup',hold=.6)

        def path(i):
            x,u,z=map(float,(xs[i],us[i],np.clip(zs[i],-5,5)))
            return np.array([den.c2p(x,0),cdf.c2p(x,0),cdf.c2p(x,u),pit.c2p(0,u),normal.c2p(z,u),normal.c2p(z,0),score.c2p(z,0)])
        def trace(i,duration):
            """One slow observation with its return, PIT and Normal score."""
            destination=i/(n-1)
            if i:
                old_position=timeline.position
                self.play(Transform(forecast_mob,forecast(i) if empirical else laws[i]),UpdateFromAlphaFunc(timeline,
                    lambda mob,a:mob.seek(old_position+(destination-old_position)*a,i)),run_time=.18 if i<3 else .10,rate_func=smooth)
            else:timeline.seek(destination,i)
            points=path(i)
            dot=PITParticle().seek(points[0],0).set_z_index(5)
            def number(value):
                return (r'-\infty' if value<0 else r'+\infty') if not np.isfinite(value) else f'{value:.3f}'
            readout=VGroup(
                eq(r'r_t='+number(xs[i])+(r'\%' if empirical else ''),23).move_to([-4.72,3.39,0]),
                eq(r'u_t='+number(us[i]),23).move_to([-.2,3.39,0]),
                eq(r'z_t='+number(zs[i]),23).move_to([4.98,3.39,0]))
            self.remove(legend);self.add(readout)
            guides=VGroup(*[DashedLine(points[k],points[k+1],color=MUTED,stroke_width=.7,dash_length=.065,stroke_opacity=.35) for k in range(1,6)])
            self.add(guides,dot)
            def tracked(mob,a):
                position,pulse,leg=flight_state(points,a)
                dot.seek(position,pulse)
                readout[1].set_opacity(1 if leg>=2 else .12)
                readout[2].set_opacity(1 if leg>=4 else .12)
            self.play(UpdateFromAlphaFunc(VGroup(dot,readout),tracked),run_time=duration,rate_func=linear)
            self.play(Transform(hist,histograms(i+1)),FadeOut(dot),FadeOut(guides),run_time=.10,rate_func=smooth)
            self.play(FadeOut(readout),FadeIn(legend),run_time=.2)
        teaching_count=1 if empirical else 2
        for i in range(teaching_count):
            trace(i,2.8)

        if empirical:
            # A stream of ~100 real dated observations (incl. the worst and best day), overlapping in flight.
            indices=np.unique(np.r_[np.linspace(1,n-1,96,dtype=int),np.argmin(xs),np.argmax(xs)])
            indices=indices[indices>0]
            interval,flight=3/60,1.6
            starts=np.arange(len(indices))*interval;duration=float(starts[-1]+flight)
            paths=[path(int(i)) for i in indices]
            laws=[forecast(int(i)) for i in indices]
            ghosts=VGroup(*[law[0].copy().set_stroke(width=.65,opacity=0) for law in laws])
            size=int(np.ceil(flight/interval))+2
            pool=VGroup(*[PITParticle(radius=.035,color=INK).set_opacity(0) for _ in range(size)])
            trails=VGroup(*[Line([0,0,0],[.001,0,0],color=INK,stroke_width=1.8).set_opacity(0) for _ in range(size)])
            self.add(ghosts,trails,pool)
            last_law=[-1]
            timeline.compact=True
            def cloud(mob,alpha):
                elapsed=alpha*duration
                active=np.flatnonzero((starts<=elapsed)&(starts+flight>elapsed))
                for dot in pool:dot.set_opacity(0)
                for trail in trails:trail.set_opacity(0)
                for ghost in ghosts:ghost.set_stroke(opacity=0)
                for dot,trail,j in zip(pool,trails,active):
                    phase=(elapsed-starts[j])/flight
                    position,pulse,leg=flight_state(paths[j],phase)
                    dot.seek(position,pulse,opacity=.68 if j!=active[-1] else 1)
                    prior=flight_state(paths[j],max(0,phase-1/(60*flight)))[0]
                    if np.linalg.norm(position-prior)>1e-6:
                        trail.put_start_and_end_on(prior,position).set_opacity(.25)
                    if leg<=2:ghosts[j].set_stroke(opacity=.16)
                spawned=int(np.searchsorted(starts,elapsed,side='right')-1)
                if spawned>=0 and spawned!=last_law[0]:
                    forecast_mob.become(laws[spawned]);last_law[0]=spawned
                if spawned>=0:
                    progress=float(np.interp(elapsed,starts,indices))/(n-1)
                    timeline.seek(progress,int(indices[spawned]))
                completed=np.flatnonzero(starts+flight<=elapsed)
                count=int(indices[completed[-1]])+1 if len(completed) else 1
                next_count=int(indices[min(len(completed),len(indices)-1)])+1
                arrival=float(starts[len(completed)-1]+flight) if len(completed) else flight
                blend=float(np.clip((elapsed-arrival)/interval,0,1)) if len(completed) else 0
                hist.become(histograms(count))
                if next_count>count and blend>0:
                    hist_target=histograms(next_count)
                    for bar,target_bar in zip(hist,hist_target):bar.interpolate(bar.copy(),target_bar,blend)
            self.play(UpdateFromAlphaFunc(VGroup(ghosts,trails,pool,hist,timeline,forecast_mob),cloud),run_time=duration,rate_func=linear)
            self.remove(pool,trails,ghosts)
        else:
            # Every remaining simulated observation, overlapping in flight.
            interval,flight=4/60,1.6
            starts=np.arange(n-teaching_count)*interval;duration=float(starts[-1]+flight)
            paths=[path(i) for i in range(teaching_count,n)]
            pool=VGroup(*[PITParticle(radius=.033).set_opacity(0) for _ in range(int(flight/interval)+2)])
            ghosts=VGroup(*[laws[0][0].copy().set_stroke(width=.65,opacity=0) for _ in pool])
            self.add(ghosts,pool)
            def stream(mob,alpha):
                elapsed=alpha*duration
                active=np.flatnonzero((starts<=elapsed)&(starts+flight>elapsed))
                for dot in pool:dot.set_opacity(0)
                for ghost in ghosts:ghost.set_stroke(opacity=0)
                for dot,ghost,j in zip(pool,ghosts,active):
                    position,pulse,leg=flight_state(paths[j],(elapsed-starts[j])/flight)
                    dot.seek(position,pulse,opacity=.65 if j!=active[-1] else 1)
                    if leg<=2:
                        ghost.set_points(laws[j+teaching_count][0].get_points().copy()).set_stroke(opacity=.10)
                step=min(n-1,teaching_count+int(elapsed/interval))
                blend=min(1,max(0,elapsed/interval-int(elapsed/interval))) if step<n-1 else 0
                for current,previous,following in zip(forecast_mob.family_members_with_points(),
                        laws[step].family_members_with_points(),laws[min(n-1,step+1)].family_members_with_points()):
                    current.interpolate(previous,following,blend)
                timeline.seek(min(1,(teaching_count+elapsed/interval)/(n-1)),step)
                completed=max(0,(elapsed-flight)/interval+1) if elapsed>=flight else 0
                count=min(n,teaching_count+int(completed))
                hist.become(histograms(count,min(1,completed-int(completed))))
            self.play(UpdateFromAlphaFunc(VGroup(pool,hist,timeline,forecast_mob,ghosts),stream),run_time=duration,rate_func=linear)
            self.remove(pool,ghosts)
        hist.become(histograms(n));timeline.compact=False;timeline.seek(1,n-1)
        self.pit_state=dict(normal=normal,phi=phi,zs=zs,legend=legend)
        self.finish('empirical_pit' if empirical else 'correct_pit',hold=.75)

    def qq_from_pit(self,empirical):
        """The Normal-score points from the PIT animation unfold into a Q-Q plot."""
        state=self.pit_state; zs=np.sort(state["zs"]); n=len(zs)
        theoretical=norm.ppf((np.arange(n)+.5)/n)
        source=state["normal"]; curve=state["phi"]
        old_chrome=self.chrome
        legend=state['legend'].copy().set_opacity(1)
        self.remove(source,curve,old_chrome,state['legend'])
        rest=Group(*self.mobjects);self.clear();self.add(rest,source,curve,legend)
        self.frame("Empirical results" if empirical else "Methods","FHS-GJR · Crisis Q–Q · 2018–2021" if empirical else "Normal Q–Q",clear=False)
        self.add(old_chrome)
        particles=VGroup(*[Dot(source.c2p(float(np.clip(z,-5,5)),float(norm.cdf(z))),radius=.022,color=INK,fill_opacity=.45) for z in zs])
        focus=VGroup(source,curve,particles)
        self.add(focus)
        self.play(FadeOut(rest),self.change_header(old_chrome),
            Succession(Wait(.15),FadeOut(legend,run_time=.75,rate_func=linear)),
            focus.animate.scale(1.25).move_to([0,.1,0]),run_time=.9,rate_func=smooth)
        xlim=QQ_X_LIMIT if empirical else 5
        ylim=QQ_Y_LIMIT if empirical else 5
        ax=axes([-xlim,xlim,1],[-ylim,ylim,1],6.3,5.8 if empirical else 6.3,-3.15,-2.8 if empirical else -3.05)
        ax.x_axis.set_opacity(0);ax.y_axis.set_opacity(0)
        diagonal=Line(ax.c2p(-xlim,-xlim),ax.c2p(xlim,xlim),color=MUTED,stroke_width=1.4)
        target=qq_cloud(ax,theoretical,zs,empirical=empirical,y_limit=ylim)
        grid=VGroup();ticks=VGroup()
        for v in ((-4,-2,0,2,4) if empirical else (-5,-3,0,3,5)):
            grid.add(Line(ax.c2p(v,-ylim),ax.c2p(v,ylim),color=RULE,stroke_width=.7))
            ticks.add(label(str(v),17,MUTED).next_to(ax.c2p(v,-ylim),DOWN,buff=.35 if empirical else .1))
        for v in ((-12,-8,-4,0,4,8,12) if empirical else (-5,-3,0,3,5)):
            grid.add(Line(ax.c2p(-xlim,v),ax.c2p(xlim,v),color=RULE,stroke_width=.7))
            ticks.add(label(str(v),17,MUTED).next_to(ax.c2p(-xlim,v),LEFT,buff=.1))
        if empirical:grid.add(qq_infinity_rows(ax,size=17))
        self.play(FadeOut(source),Transform(curve,diagonal),FadeIn(grid),FadeIn(ticks),Transform(particles,target),run_time=2.2,rate_func=smooth)
        self.remove(focus,curve);self.add(grid,ticks,diagonal,particles)
        labels=VGroup(label("Theoretical Normal quantiles",22).move_to([0,-3.65,0]),
            label("Ordered Normal scores",21).rotate(PI/2).move_to([-4.05,.1,0]))
        self.reveal(labels)
        if empirical:
            below=int(np.isneginf(zs).sum());above=int(np.isposinf(zs).sum())
            self.reveal(label(f'Outside support: {below} below / {above} above',17,MUTED).move_to([0,-4.04,0]))
        self.qq_state=dict(ax=ax,points=particles,expected=theoretical,empirical=empirical)
        self.finish("empirical_qq" if empirical else "correct_qq")

    def qq_tails(self):
        """Same Q-Q axes; the points move to Student-t(5) quantiles to show heavy tails."""
        state=self.qq_state;expected=state["expected"];ax=state["ax"];n=len(expected)
        old_chrome=self.chrome
        self.frame("Methods","Tail misspecification",clear=False)
        self.add(old_chrome)
        scores=t.ppf((np.arange(n)+.5)/n,df=5)*t.pdf(0,df=5)/norm.pdf(0)
        target=VGroup(*[Dot(ax.c2p(float(x),float(y)),radius=.037 if abs(x)>1.645 else .021,color=NAVY if abs(x)>1.645 else INK) for x,y in zip(expected,scores)])
        self.play(self.change_header(old_chrome),Transform(state["points"],target),run_time=2.0,rate_func=smooth)
        callouts=VGroup(label("Heavier losses",26,NAVY).move_to([-5.5,-1.4,0]),label("Heavier gains",26,NAVY).move_to([5.3,2.0,0]))
        self.play(FadeIn(callouts),run_time=.6,rate_func=smooth)
        self.finish("tail_qq")
