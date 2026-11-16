<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<svelte:head>
	<title>Perspektivische Analyse</title>
</svelte:head>

{@render children()}
