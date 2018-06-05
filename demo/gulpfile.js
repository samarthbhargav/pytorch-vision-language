var gulp = require('gulp');
var plugin = require('gulp-load-plugins')();
var config = {
    "bower" : "bower_components",
    "src" : "public/src",
    "dest" : "public/dist",
    "debug" : !(process.env.APPLICATION_ENVIRONMENT === 'production')
};

gulp.task('bower', function (cb) {
    require('child_process').exec('bower-installer', function (err, stdout, stderr) {
        console.log(stdout);
        console.log(stderr);
        cb(err);
    });
})

gulp.task('lint', function() {
    return gulp.src([
        config.src + '/gve/**/*.js',
        './routes/**/*.js'
        ])
    .pipe(plugin.jshint())
    .pipe(plugin.jshint.reporter('default'));
});

gulp.task('css', function() {
    return gulp.src([
        config.src + '/lib/material-design-lite/material.css',
        config.src + '/lib/material-design-lite/material.grey-pink.min.css',
        config.src + '/lib/material-design-lite/roboto.css',
        config.src + '/lib/material-design-lite/material-icons.css'
    ])
    .pipe(plugin.if(config.debug, plugin.sourcemaps.init()))
    .pipe(plugin.cleanCss())
    .pipe(plugin.autoprefixer({
        'browsers' : ['last 2 versions'],
        'cascade' : false
    }))
    .pipe(plugin.concat('lib.css'))
    .pipe(plugin.if(config.debug, plugin.sourcemaps.write()))
    .pipe(gulp.dest(config.dest + '/css'))
    .pipe(plugin.if(config.debug, plugin.livereload()));
});

gulp.task('sass', function() {
    return gulp.src([
        config.src + '/lib/getmdl-select/src/scss/getmdl-select.scss',
        config.src + '/gve/**/*.scss',
        '!' + config.src + '/gve/common/variables.scss',
        '!' + config.src + '/gve/common/mixins.scss'
    ])
    .pipe(plugin.if(config.debug, plugin.sourcemaps.init()))
    .pipe(plugin.sass())
    .pipe(plugin.autoprefixer({
        'browsers' : ['last 2 versions'],
        'cascade' : false
    }))
    .pipe(plugin.cleanCss())
    .pipe(plugin.rename('main.css'))
    .pipe(plugin.if(config.debug, plugin.sourcemaps.write()))
    .pipe(gulp.dest(config.dest + '/css'))
    .pipe(plugin.if(config.debug, plugin.livereload()));
});

gulp.task('markup', function() {
    return gulp.src(['./public/index.html', config.src + '/**/*.html'])
    .pipe(plugin.if(config.debug, plugin.livereload()));
});

gulp.task('scripts', function() {
  return gulp.src([
        config.src + '/lib/angular/angular.js',
        config.src + '/lib/material-design-lite/material.js',
        config.src + '/lib/getmdl-select/src/js/getmdl-select.js',
        config.src + '/gve/*.js',
        config.src + '/gve/**/*.js'
        ])
    .pipe(plugin.if(config.debug, plugin.sourcemaps.init()))
    .pipe(plugin.concat('main.js'))
    .pipe(plugin.uglify())
    .pipe(plugin.if(config.debug, plugin.sourcemaps.write()))
    .pipe(gulp.dest(config.dest + '/js'))
    .pipe(plugin.if(config.debug, plugin.livereload()));
});

gulp.task('watch', ['express'], function () {
    plugin.livereload.listen();
    gulp.watch(config.src + '/**/*.js', ['lint', 'scripts']);
    gulp.watch(config.src + '/**/*.scss', ['sass']);
    gulp.watch(['./public/index.html', config.src + '/**/*.html'], ['markup']);
});

gulp.task('express', ['lint', 'sass', 'css', 'scripts'], function () {
    var express = require('express');
    var app = express();
    app.use(express.static(__dirname + '/public/dist'), require('./routes/routes')(express));
    app.get('/', function (req, res) {
        res.sendFile(__dirname + '/public/index.html');
    });
    app.listen(8080);
});

gulp.task('default', function () {
    return (config.debug) ? ['watch'] : ['express'];
}());
